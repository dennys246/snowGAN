import os, atexit, re, shutil, math
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from glob import glob

from snowgan.losses import compute_gradient_penalty
from snowgan.generate import generate, make_movie
from snowgan.log import save_history, load_history
from snowgan.data.dataset import DataManager
from snowgan.utils import compute_fade_alpha
from snowgan.augment import augment as diff_augment
 
class Trainer:

    def __init__(self, generator, discriminator):
        
        # Generator and discriminator models
        self.gen = generator

        print(f"Initializing trainer in cwd {os.getcwd()} with checkpoint {self.gen.config.checkpoint}...")
        
        # If model not yet built
        #if not self.gen.built:
        #    self.gen.build(self.gen.config.resolution)

        # Attempt to load weights if they haven't been built yet
        if os.path.exists(self.gen.config.checkpoint):
            try:
                # Load weights from .keras file
                self.gen.model.load_weights(self.gen.config.checkpoint)
                print(f"Generator weights loaded successfully from {self.gen.config.checkpoint}")
            except Exception as e:
                print(f"Warning: could not load generator weights: {e}. Using freshly initialized weights.")
            # Load fade endpoints weights (toRGB_prev) for mid-fade resume
            if getattr(self.gen, 'fade_endpoints', None) is not None:
                fade_weights_path = os.path.join(os.path.dirname(self.gen.config.checkpoint), "generator_fade_endpoints.weights.h5")
                if os.path.exists(fade_weights_path):
                    try:
                        self.gen.fade_endpoints.load_weights(fade_weights_path)
                        print(f"Generator fade endpoints loaded from {fade_weights_path}")
                    except Exception as e:
                        print(f"Warning: could not load fade endpoint weights: {e}. Using freshly initialized weights.")
        else:
            print("Generator saved weights not found, new model initialized")


        self.disc = discriminator

        # If model not yet built
        #if not self.disc.built:
        #    self.disc.build(self.disc.config.resolution)

        # If weights haven't been initialized
        if os.path.exists(self.disc.config.checkpoint):
            try:
                self.disc.model.load_weights(self.disc.config.checkpoint)
                print(f"Discriminator weights loaded successfully from {self.disc.config.checkpoint}")
            except Exception as e:
                print(f"Warning: could not load discriminator weights: {e}. Using freshly initialized weights.")
        else:
            print("Disciminator saved weights not found, new model initialized")

        self.save_dir = self.gen.config.save_dir # Save dictory for the model and it's generated images

        self.batch_size = self.gen.config.batch_size # Number of images to load in per training batch

        self.n_samples = self.gen.config.n_samples # Number of synthetic images to generate after training
        cleanup_raw = getattr(self.gen.config, "cleanup_milestone", 1000)
        try:
            cleanup_value = int(cleanup_raw)
        except (TypeError, ValueError):
            cleanup_value = 1000
        self.cleanup_milestone = max(0, cleanup_value)
        if hasattr(self.gen.config, "cleanup_milestone"):
            self.gen.config.cleanup_milestone = self.cleanup_milestone
        if hasattr(self.disc.config, "cleanup_milestone"):
            self.disc.config.cleanup_milestone = self.cleanup_milestone
        self._save_interval = self.cleanup_milestone if self.cleanup_milestone > 0 else 1000

        self.dataset = DataManager(self.gen.config)
        self.train_ind = self.gen.config.train_ind
        self.trained_data = []

        self.loss = {'disc': [], 'gen': []}

        self.loss, self.trained_data = load_history(self.gen.config.save_dir) # Load any history we can find in save directory

        atexit.register(self.save_model)

        print("Trainer initialized...")
        # Global step for fade scheduling (persisted in config)
        gen_step = int(getattr(self.gen.config, 'fade_step', 0) or 0)
        disc_step = int(getattr(self.disc.config, 'fade_step', 0) or 0)
        self.global_step = max(gen_step, disc_step)
        self.fade_steps = max(1, int(getattr(self.gen.config, 'fade_steps', 1)))
        self.fade_complete = not (getattr(self.gen.config, 'fade', False) and getattr(self.gen, 'fade_endpoints', None) is not None)
        if self.global_step >= self.fade_steps:
            self.fade_complete = True
        # Keep generator/discriminator configs aligned without forcing a disk write on init
        self._sync_fade_progress(persist=False)

        # --- Post-progressive training improvements ---
        # When spectral norm is active, reduce gradient penalty weight since SN
        # already enforces the Lipschitz constraint — heavy GP on top over-constrains
        # the discriminator and causes color drift / mode collapse.
        use_sn = getattr(self.disc.config, 'spectral_norm', False)
        if use_sn and self.disc.config.lambda_gp > 1.0:
            old_gp = self.disc.config.lambda_gp
            self.disc.config.lambda_gp = 1.0
            print(f"Spectral norm active: reduced lambda_gp from {old_gp} to {self.disc.config.lambda_gp}")

        # Differentiable augmentation
        self.use_augment = getattr(self.gen.config, 'augment', False)

        # Learning rate decay (cosine annealing)
        self.lr_decay = getattr(self.gen.config, 'lr_decay', None)
        self.lr_min = getattr(self.gen.config, 'lr_min', 1e-7)
        self.gen_lr_base = float(self.gen.config.learning_rate)
        self.disc_lr_base = float(self.disc.config.learning_rate)

        # Generator EMA (exponential moving average) shadow weights
        self.ema_decay = getattr(self.gen.config, 'ema_decay', 0.0)
        self.ema_weights = None
        if self.ema_decay > 0:
            self._init_ema()

        # FID-based checkpointing
        self.fid_interval = getattr(self.gen.config, 'fid_interval', 0)
        self.best_fid = float('inf')

        # Gradient clipping
        self.grad_clip_norm = getattr(self.gen.config, 'grad_clip_norm', 0.0)

        # Fixed seed for consistent visual tracking across batches
        self._tracking_seed = tf.random.normal([self.n_samples, self.gen.config.latent_dim])

        # Adaptive disc/gen step ratio
        self.adaptive_steps = getattr(self.gen.config, 'adaptive_steps', False)
        self._base_disc_steps = self.disc.config.training_steps
        self._disc_loss_ema = 0.0
        self._gen_loss_ema = 0.0

        # Adaptive augmentation (ADA) — adjust augment probability based on disc overfitting
        self.ada_target = getattr(self.gen.config, 'ada_target', 0.0)
        self.augment_p = 0.5 if self.use_augment else 0.0

        # Multi-scale discriminator
        self.multiscale_disc = getattr(self.disc.config, 'multiscale_disc', False)
        self.disc_lowres = None
        self.disc_lowres_optimizer = None
        if self.multiscale_disc:
            self._build_lowres_disc()
            lowres_path = os.path.join(os.path.dirname(self.disc.config.checkpoint), "discriminator_lowres.keras")
            if os.path.exists(lowres_path):
                try:
                    self.disc_lowres.load_weights(lowres_path)
                    print(f"Low-res discriminator loaded from {lowres_path}")
                except Exception as e:
                    print(f"Warning: could not load low-res disc weights: {e}. Initialized fresh.")

    def _init_ema(self):
        """Initialize EMA shadow weights as a copy of current generator weights on CPU to save VRAM."""
        with tf.device('/CPU:0'):
            self.ema_weights = [tf.Variable(w, trainable=False, name=f"ema/{w.name}")
                                for w in self.gen.model.trainable_variables]
        # Load persisted EMA weights if available
        ema_path = os.path.join(os.path.dirname(self.gen.config.checkpoint), "generator_ema.weights.h5")
        if os.path.exists(ema_path):
            try:
                # Build a temporary model clone to load weights
                tmp_model = keras.models.clone_model(self.gen.model)
                tmp_model.build(self.gen.model.input_shape)
                tmp_model.load_weights(ema_path)
                for ema_var, tmp_var in zip(self.ema_weights, tmp_model.trainable_variables):
                    ema_var.assign(tmp_var)
                del tmp_model
                print(f"EMA weights loaded from {ema_path}")
            except Exception as e:
                print(f"Warning: could not load EMA weights: {e}. Using current generator weights.")

    def _update_ema(self):
        """Update EMA shadow weights with current generator weights."""
        if self.ema_weights is None:
            return
        decay = self.ema_decay
        for ema_var, model_var in zip(self.ema_weights, self.gen.model.trainable_variables):
            ema_var.assign(decay * ema_var + (1.0 - decay) * model_var)

    def _apply_ema_to_generator(self):
        """Swap EMA weights into the generator (for generation/eval)."""
        if self.ema_weights is None:
            return None
        # Save current weights so we can restore later
        backup = [tf.identity(w) for w in self.gen.model.trainable_variables]
        for model_var, ema_var in zip(self.gen.model.trainable_variables, self.ema_weights):
            model_var.assign(ema_var)
        return backup

    def _restore_generator_weights(self, backup):
        """Restore generator weights from a backup (after EMA swap)."""
        if backup is None:
            return
        for model_var, bak in zip(self.gen.model.trainable_variables, backup):
            model_var.assign(bak)

    def _update_learning_rates(self):
        """Apply cosine annealing LR decay based on post-fade step count."""
        if self.lr_decay != "cosine":
            return
        # Only decay after fade completes; use steps since fade ended
        if not self.fade_complete:
            return
        post_fade_step = self.global_step - self.fade_steps
        if post_fade_step < 0:
            post_fade_step = 0
        # Cosine decay over a long horizon (200k steps) with minimum floor
        decay_steps = 200000
        progress = min(post_fade_step / decay_steps, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        gen_lr = self.lr_min + (self.gen_lr_base - self.lr_min) * cosine_factor
        disc_lr = self.lr_min + (self.disc_lr_base - self.lr_min) * cosine_factor

        self.gen.optimizer.learning_rate.assign(gen_lr)
        self.disc.optimizer.learning_rate.assign(disc_lr)

    def _build_lowres_disc(self):
        """Build a lightweight discriminator for 256x256 input (multi-scale feedback)."""
        use_sn = getattr(self.disc.config, 'spectral_norm', False)
        inputs = keras.Input(shape=(256, 256, 3))
        x = inputs
        for filters in [64, 128, 256]:
            conv = keras.layers.Conv2D(filters, 3, strides=2, padding='same')
            x = keras.layers.SpectralNormalization(conv)(x) if use_sn else conv(x)
            x = keras.layers.LeakyReLU(negative_slope=0.25)(x)
        x = keras.layers.Flatten()(x)
        dense = keras.layers.Dense(1)
        outputs = keras.layers.SpectralNormalization(dense)(x) if use_sn else dense(x)
        self.disc_lowres = keras.Model(inputs, outputs, name="DiscriminatorLowRes")
        self.disc_lowres_optimizer = keras.optimizers.Adam(
            learning_rate=self.disc.config.learning_rate,
            beta_1=self.disc.config.beta_1,
            beta_2=self.disc.config.beta_2
        )
        print(f"Multi-scale discriminator built (256x256, {self.disc_lowres.count_params()} params)")

    def _update_adaptive_steps(self, disc_loss, gen_loss):
        """Adjust disc training steps based on loss balance."""
        if not self.adaptive_steps:
            return
        # Exponential moving average of losses
        alpha = 0.05
        self._disc_loss_ema = alpha * abs(float(disc_loss)) + (1 - alpha) * self._disc_loss_ema
        self._gen_loss_ema = alpha * abs(float(gen_loss)) + (1 - alpha) * self._gen_loss_ema
        total = self._disc_loss_ema + self._gen_loss_ema + 1e-8
        disc_ratio = self._disc_loss_ema / total

        # Adjust every 100 steps
        if self.global_step % 100 == 0 and self.global_step > 0:
            max_steps = self._base_disc_steps * 2
            if disc_ratio < 0.3:
                # Disc too weak — give it more steps
                self.disc.config.training_steps = min(self.disc.config.training_steps + 1, max_steps)
                print(f"Adaptive steps: disc weak (ratio={disc_ratio:.2f}), disc_steps -> {self.disc.config.training_steps}")
            elif disc_ratio > 0.7:
                # Disc too strong — reduce its steps
                self.disc.config.training_steps = max(self.disc.config.training_steps - 1, 1)
                print(f"Adaptive steps: disc strong (ratio={disc_ratio:.2f}), disc_steps -> {self.disc.config.training_steps}")

    def _update_ada(self, real_scores):
        """Adjust augmentation probability based on discriminator overfitting (ADA)."""
        if self.ada_target <= 0:
            return
        # rt = fraction of real images the disc correctly identifies (score > 0 in WGAN)
        rt = float(tf.reduce_mean(tf.cast(real_scores > 0, tf.float32)))
        # Adjust p every 50 steps
        if self.global_step % 50 == 0 and self.global_step > 0:
            if rt > self.ada_target:
                # Disc overfitting — increase augmentation
                self.augment_p = min(self.augment_p + 0.01, 0.8)
            else:
                # Disc not overfitting — decrease augmentation
                self.augment_p = max(self.augment_p - 0.01, 0.0)

    def _compute_fid(self, num_samples=64):
        """
        Compute a lightweight FID approximation using InceptionV3 features.
        Compares real dataset samples against generated samples.
        Returns FID score or None if computation fails.
        """
        try:
            from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

            # Load inception model on demand — do NOT cache in VRAM permanently
            inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

            def get_features_batched(images, batch_size=4):
                """Extract features in small batches to avoid OOM at high resolutions."""
                all_features = []
                for i in range(0, len(images), batch_size):
                    chunk = images[i:i + batch_size]
                    chunk = tf.image.resize(chunk, (299, 299))
                    chunk = (chunk + 1.0) * 127.5
                    chunk = preprocess_input(chunk)
                    feats = inception_model(chunk, training=False)
                    all_features.append(feats.numpy())
                return np.concatenate(all_features, axis=0)

            # Generate fake samples and extract features per-chunk to avoid
            # holding all full-resolution images in memory simultaneously.
            backup = self._apply_ema_to_generator()
            fake_features_list = []
            gen_batch = 4
            for i in range(0, num_samples, gen_batch):
                n = min(gen_batch, num_samples - i)
                noise = tf.random.normal([n, self.gen.config.latent_dim])
                chunk = self.gen.model(noise, training=False)
                chunk = tf.image.resize(chunk, (299, 299))
                chunk = (chunk + 1.0) * 127.5
                chunk = preprocess_input(chunk)
                fake_features_list.append(inception_model(chunk, training=False).numpy())
            self._restore_generator_weights(backup)

            # Get real samples WITHOUT advancing the training pointer
            saved_ind = self.gen.config.train_ind
            real_images = self.dataset.batch(num_samples, 'magnified_profile')
            self.gen.config.train_ind = saved_ind  # Restore pointer
            if real_images is None:
                return None

            fake_features = np.concatenate(fake_features_list, axis=0)
            real_features = get_features_batched(real_images)

            # Compute FID: ||mu_r - mu_f||^2 + Tr(C_r + C_f - 2*sqrt(C_r @ C_f))
            mu_r, mu_f = np.mean(real_features, axis=0), np.mean(fake_features, axis=0)
            sigma_r = np.cov(real_features, rowvar=False)
            sigma_f = np.cov(fake_features, rowvar=False)

            diff = mu_r - mu_f
            # Stable matrix sqrt via eigendecomposition
            product = sigma_r @ sigma_f
            eigenvalues, _ = np.linalg.eigh(product)
            eigenvalues = np.maximum(eigenvalues, 0)
            sqrt_trace = np.sum(np.sqrt(eigenvalues))

            fid = float(np.dot(diff, diff) + np.trace(sigma_r) + np.trace(sigma_f) - 2 * sqrt_trace)

            # Free InceptionV3 from VRAM — it's only needed during FID computation
            del inception_model

            return fid
        except Exception as e:
            print(f"Warning: FID computation failed: {e}")
            return None

    def train(self, batch_size = 8, epochs = 1):
        """
        Initializes training the discriminator and generator based on requested
        runtime behavioral. The model is saved after every training batch to preserve
        training history. 
        
        Function arguments:
            batches (int or None) - Number of training batches to learn from before stopping
            batch_size (int) - Number of images to load into each training batch
            epochs (int or None) - Number of epochs to train per batch, defaults to class default
        """

        # Update hyperparameters if passed in before training
        if batch_size: self.batch_size = batch_size

        # Iterate through requested training batches
        for epoch in range(epochs):
            batch = 1

            batched_images = glob(f"{self.gen.config.save_dir}/synthetic_images/*batch*.png")
            for batch_image in batched_images:
                batch_number = int(batch_image.split('batch_')[1].split('_')[0])
                if batch_number >= batch:
                    batch = batch_number + 1

            trainable_data = True
            while trainable_data:
                # Load a new batch of subjects
                print(f"Grab a batch of data")
                x = self.dataset.batch(batch_size, 'magnified_profile') 
                if x is None:
                    self.gen.config.train_ind = 390
                    x = self.dataset.batch(batch_size, 'magnified_profile') 
                    
                print(f"Data batched")
                if x is None:
                    print(f"Training Epoch Complete")
                    break

                print(f"Training on batch {batch}...")

                self.train_step(x) # Train on batch of images
                print(f'Epoch {epoch} | Batch {batch} | Generator loss: {round(float(self.loss["gen"][-1]), 3)} | Discrimintator loss: {round(float(self.loss["disc"][-1]), 3)} |')
                
                # Plot and generate synthetic images every 10 batches to reduce I/O overhead
                if batch % 10 == 0:
                    self.plot_history()

                    if self.n_samples:
                        backup = self._apply_ema_to_generator()
                        _ = generate(self.gen, count = self.n_samples, seed_size = self.gen.config.latent_dim, save_dir = f"{self.save_dir}/synthetic_images/", filename_prefix = f'batch_{batch}_synthetic', seed = self._tracking_seed)
                        self._restore_generator_weights(backup)

                # FID-based best model checkpointing
                if self.fid_interval > 0 and self.global_step % self.fid_interval == 0:
                    fid = self._compute_fid()
                    if fid is not None:
                        print(f"FID @ step {self.global_step}: {fid:.2f} (best: {self.best_fid:.2f})")
                        if fid < self.best_fid:
                            self.best_fid = fid
                            best_dir = os.path.join(self.save_dir, "best_fid/")
                            self.save_model(best_dir)
                            print(f"New best FID! Model saved to {best_dir}")

                # Save the models state
                if batch % self._save_interval == 0:
                    self.save_model(f"{self.save_dir}/batch_{batch}/")
                    if self.cleanup_milestone > 0 and batch % self.cleanup_milestone == 0:
                        self._cleanup_saved_batches(100)
                
                batch += 1

    def train_step(self, images, disc_steps = None, gen_steps = None):
        """
        This function trains the snowGAN on the passed in images, with variables 
        training length of the discriminator and generator. This function implements
        WGAN-GP unique EMD loss and gradient penalty regularization.
        
        Function arguments:
            images (4D numpy array) - numpy array of RGB images set to the class resolution (sample axis = 0)
            disc_steps (int) - Number N of training steps for the discriminator per epoch
            gen_steps (int) - Number M of training steps for the generator per epoch
        """
        # Update training parameters if passed in
        if disc_steps:
            self.disc.config.training_steps = disc_steps
        if gen_steps:
            self.gen.config.training_steps = gen_steps

        # Set batch size to the number of images passed in
        batch_size = tf.shape(images)[0] 

        # Generate noise for generator input
        noise = tf.random.normal([batch_size, self.gen.config.latent_dim])
        print(f"Noise shape: {noise.shape}")
        
        # Current augmentation probability (ADA adjusts this dynamically)
        aug_p = self.augment_p

        # Train the discriminator N times
        real_scores = None
        for _ in range(self.disc.config.training_steps):
            # Generate synthetic images outside the tape — no need to track generator ops for disc training
            synthetic_images = tf.stop_gradient(self._generate_with_fade(noise, training=True))

            # Apply augmentation outside the tape to avoid storing intermediate tensors in memory
            disc_real = diff_augment(images, p=aug_p) if self.use_augment else images
            disc_fake = diff_augment(synthetic_images, p=aug_p) if self.use_augment else synthetic_images

            # Main discriminator tape — separate from lowres to reduce peak VRAM
            with tf.GradientTape() as tape:
                # Forward propogate real & synethic images to the discriminator
                output = self.disc.model(disc_real, training=True)
                synthetic_output = self.disc.model(disc_fake, training=True)
                real_scores = output  # Track for ADA

                # Calculate discriminators gradient penalty (on unaugmented images for stable gradients)
                gp = compute_gradient_penalty(self.disc, images, synthetic_images)

                # Calculate EMD/loss for the discriminators outputs
                disc_loss = self.disc.get_loss(output, synthetic_output, gp, self.disc.config.lambda_gp)

            # Backpropogate main discriminator
            disc_gradients = tape.gradient(disc_loss, self.disc.model.trainable_variables)
            if self.grad_clip_norm > 0:
                disc_gradients, _ = tf.clip_by_global_norm(disc_gradients, self.grad_clip_norm)
            self.disc.optimizer.apply_gradients(zip(disc_gradients, self.disc.model.trainable_variables))

            # Free main disc tape activations before lowres pass
            del tape, disc_gradients

            # Multi-scale discriminator in its own tape (separate VRAM peak)
            if self.disc_lowres is not None:
                real_lowres = tf.image.resize(images, (256, 256))
                fake_lowres = tf.image.resize(synthetic_images, (256, 256))
                with tf.GradientTape() as lr_tape:
                    real_out_lr = self.disc_lowres(real_lowres, training=True)
                    fake_out_lr = self.disc_lowres(fake_lowres, training=True)
                    disc_loss_lr = tf.reduce_mean(tf.cast(fake_out_lr, tf.float32)) - tf.reduce_mean(tf.cast(real_out_lr, tf.float32))
                lowres_grads = lr_tape.gradient(disc_loss_lr, self.disc_lowres.trainable_variables)
                if self.grad_clip_norm > 0:
                    lowres_grads, _ = tf.clip_by_global_norm(lowres_grads, self.grad_clip_norm)
                self.disc_lowres_optimizer.apply_gradients(zip(lowres_grads, self.disc_lowres.trainable_variables))
                disc_loss = disc_loss + 0.5 * disc_loss_lr  # For logging
                del lr_tape, lowres_grads, real_lowres, fake_lowres

            # Free augmented copies before gen training
            del disc_real, disc_fake

        # Train the generator M times
        for _ in range(self.gen.config.training_steps):
            # Generate random noise to pass into the generator
            noise = tf.random.normal([batch_size, self.gen.config.latent_dim])
            # Initialize automtic differentiation during forward propogation
            with tf.GradientTape() as tape:
                # Forward pass noise through the generator to create synthetic images
                use_fade = self._use_fade()
                synthetic_images = self._generate_with_fade(noise, training=True)
                # Apply differentiable augmentation to fake images for generator training
                disc_fake = diff_augment(synthetic_images, p=aug_p) if self.use_augment else synthetic_images
                # Forward pass generator outputs through the discriminator for loss calculation
                synthetic_output = self.disc.model(disc_fake, training=True)
                # Calculate generator loss for backpropogation by calculating mean of disc output
                gen_loss = self.gen.get_loss(synthetic_output)

                # Multi-scale generator loss
                if self.disc_lowres is not None:
                    fake_lowres = tf.image.resize(synthetic_images, (256, 256))
                    if self.use_augment:
                        fake_lowres = diff_augment(fake_lowres, p=aug_p)
                    fake_out_lr = self.disc_lowres(fake_lowres, training=True)
                    gen_loss = gen_loss + 0.5 * (-tf.reduce_mean(tf.cast(fake_out_lr, tf.float32)))

            # Backpropogate to calculate gradient of trainable parameters given loss
            var_list = list(self.gen.model.trainable_variables)
            # If fading, also update fade endpoint layers (e.g., toRGB_prev)
            if use_fade and hasattr(self.gen, 'fade_endpoints') and self.gen.fade_endpoints is not None:
                # Extend with unique vars from fade_endpoints by variable name (TF variables lack ref())
                existing_names = {v.name for v in var_list}
                fade_vars = [v for v in self.gen.fade_endpoints.trainable_variables if v.name not in existing_names]
                var_list.extend(fade_vars)

            gen_gradients = tape.gradient(gen_loss, var_list)
            # Apply gradient clipping if configured
            if self.grad_clip_norm > 0:
                gen_gradients, _ = tf.clip_by_global_norm(gen_gradients, self.grad_clip_norm)
            # Apply the gradients and update trainable parameters (w, b, etc.)
            self.gen.optimizer.apply_gradients(zip(gen_gradients, var_list))

        # Record loss of models to their histories
        self.loss['gen'].append(float(gen_loss))
        self.loss['disc'].append(float(disc_loss))
        # Increment global step after a full train step
        self.global_step += 1
        self._update_fade_completion()
        # Persist fade progress across generator/discriminator configs for resume support
        # Throttle disk writes to every 50 steps (atexit save_model covers final state)
        self._sync_fade_progress(persist=(self.global_step % 50 == 0))

        # Post-step improvements
        self._update_ema()
        self._update_learning_rates()
        self._update_adaptive_steps(disc_loss, gen_loss)
        if real_scores is not None:
            self._update_ada(real_scores)

    def _use_fade(self):
        return getattr(self.gen.config, 'fade', False) and getattr(self.gen, 'fade_endpoints', None) is not None and not self.fade_complete

    def _generate_with_fade(self, noise, training=True):
        if self._use_fade():
            fade_total = self.fade_steps
            fade_progress = min(self.global_step, fade_total)
            alpha = compute_fade_alpha(fade_progress, fade_total)
            prev_up, curr_img = self.gen.fade_endpoints(noise, training=training)
            alpha = tf.cast(alpha, curr_img.dtype)
            return (1.0 - alpha) * prev_up + alpha * curr_img
        return self.gen.model(noise, training=training)

    def _update_fade_completion(self):
        if not self.fade_complete and self.global_step >= self.fade_steps:
            self.fade_complete = True

    def save_model(self, path = None):
        """
        Save the currently loaded generator and discriminator to a given path

        Function arguments:
            path (str) - Folder to save the generator and discriminator .keras files
        """
        # Update path if none passed in
        if path == None:
            path = self.save_dir
        
        os.makedirs(path, exist_ok = True)
        # Persist configs alongside weights for reproducible snapshots
        self._save_configs(path)

        # Save generator and discriminator as .keras files first — weights are the
        # most important artifact and must not be blocked by a plotting failure.
        self.gen.model.save(f"{path}/generator.keras")
        self.disc.model.save(f"{path}/discriminator.keras")

        # Log and plot history (non-critical — failures here won't lose weights)
        try:
            save_history(self.gen.config.save_dir, self.loss, self.trained_data)
            self.plot_history()
        except Exception as e:
            print(f"Warning: failed to save/plot history: {e}")
        # Save fade endpoints weights so mid-fade resume preserves toRGB_prev
        if getattr(self.gen, 'fade_endpoints', None) is not None:
            self.gen.fade_endpoints.save_weights(f"{path}/generator_fade_endpoints.weights.h5")
        # Save multi-scale discriminator weights
        if self.disc_lowres is not None:
            self.disc_lowres.save(f"{path}/discriminator_lowres.keras")
        # Save EMA shadow weights for resume
        if self.ema_weights is not None:
            backup = self._apply_ema_to_generator()
            self.gen.model.save_weights(f"{path}/generator_ema.weights.h5")
            self._restore_generator_weights(backup)
        print(f"Models saved in {path}...")

    def load_model(self, path = None):
        """
        Load the generator and discriminator into the snowGAN object from the path passed in
        
        Function arguments:
            path (str) - Folder containing the generator and discriminator .keras files
        """
        # Update path if none passed in
        if path == None:
            path = self.save_dir

        # Load generator and discriminator .keras files
        self.model.gen.model = tf.keras.models.load_model(f"{path}keras/generator.keras")
        self.model.disc.model = tf.keras.models.load_model(f"{path}keras/discriminator.keras")
        print(f"Models loaded from {path}...")


    def plot_history(self):
        # Check if save folder exists yet
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir, exist_ok = True)
        # Plot the generator and discriminator loss history
        plt.plot([float(v) for v in self.loss['gen']], label = 'Generator loss')
        plt.plot([float(v) for v in self.loss['disc']], label = 'Discriminator loss')
        plt.title("GAN History")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(self.save_dir, 'history.png'))
        plt.close()

    @staticmethod
    def _extract_batch_number(path: str):
        basename = os.path.basename(os.path.normpath(path))
        match = re.search(r"batch_(\d+)", basename)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _cleanup_saved_batches(self, keep_every: int):
        """
        Remove checkpoint directories for batches that do not align with the cleanup milestone
        and trim only the seven most recent synthetic samples for those batches.
        """
        if keep_every <= 0:
            return

        removed_checkpoints = 0
        trimmed_images = 0
        synthetic_dir = os.path.join(self.save_dir, "synthetic_images")

        for checkpoint_path in glob(os.path.join(self.save_dir, "batch_*")):
            if not os.path.isdir(checkpoint_path):
                continue
            batch_number = self._extract_batch_number(checkpoint_path)
            if batch_number is None or batch_number % keep_every == 0:
                continue

            shutil.rmtree(checkpoint_path, ignore_errors=True)
            removed_checkpoints += 1

            if os.path.isdir(synthetic_dir):
                pattern = os.path.join(synthetic_dir, f"batch_{batch_number}_synthetic_*.png")
                indexed_images = []
                for image_path in glob(pattern):
                    stem = os.path.splitext(os.path.basename(image_path))[0]
                    try:
                        image_index = int(stem.split("_")[-1])
                    except (ValueError, IndexError):
                        continue
                    indexed_images.append((image_index, image_path))
                indexed_images.sort(key=lambda item: item[0])
                for _, image_path in indexed_images[-7:]:
                    try:
                        os.remove(image_path)
                        trimmed_images += 1
                    except (FileNotFoundError, ValueError):
                        continue

        if removed_checkpoints or trimmed_images:
            print(
                f"Cleanup milestone reached (keep every {keep_every} batches): "
                f"removed {removed_checkpoints} checkpoint directories, trimmed {trimmed_images} synthetic images."
            )

    def _save_configs(self, path: str):
        """
        Save generator and discriminator configs to the given snapshot path without
        permanently changing their primary config file targets.
        """
        gen_cfg_path = os.path.join(path, "generator_config.json")
        disc_cfg_path = os.path.join(path, "discriminator_config.json")

        # Preserve original config destinations so subsequent saves still hit the main paths
        gen_orig = getattr(self.gen.config, "config_filepath", None)
        disc_orig = getattr(self.disc.config, "config_filepath", None)
        try:
            self.gen.config.save_config(gen_cfg_path)
            self.disc.config.save_config(disc_cfg_path)
        except Exception as e:
            print(f"Warning: failed to save configs in {path}: {e}")
        finally:
            if gen_orig is not None:
                self.gen.config.config_filepath = gen_orig
            if disc_orig is not None:
                self.disc.config.config_filepath = disc_orig

    def _sync_fade_progress(self, persist=True):
        """
        Mirror fade progress/targets to both generator and discriminator configs so training can resume seamlessly.
        """
        target_step = int(self.global_step)
        gen_cfg = getattr(self.gen, 'config', None)
        disc_cfg = getattr(self.disc, 'config', None)

        for cfg, label in ((gen_cfg, "generator"), (disc_cfg, "discriminator")):
            if cfg is None:
                continue
            try:
                cfg.fade_step = target_step
                if gen_cfg is not None and cfg is not gen_cfg and hasattr(cfg, 'fade_steps') and hasattr(gen_cfg, 'fade_steps'):
                    if cfg.fade_steps != gen_cfg.fade_steps:
                        cfg.fade_steps = gen_cfg.fade_steps
                if persist:
                    cfg.save_config()
            except Exception as e:
                print(f"Warning: failed to persist fade configuration for {label}: {e}")


