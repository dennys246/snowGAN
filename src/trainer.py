import os, atexit
import tensorflow as tf
from matplotlib import pyplot as plt
from glob import glob

from generator import generate, make_movie

class Trainer:

    def __init__(self, save_dir, generator, discriminator, gen_config, disc_config, dataset):

        self.save_dir = save_dir # Save dictory for the model and it's generated images

        self.batch_size = gen_config['batch_size'] # Number of images to load in per training batch

        self.synthetics = gen_config['synthetics'] # Number of synthetic images to generate after training

        # Generator and discriminator models
        self.gen = generator
        self.disc = discriminator

            # Setup optimizers from config or defaults
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_config.get("gen_lr", 1e-4),
                                        beta_1=gen_config.get("gen_beta1", 0.5),
                                        beta_2=gen_config.get("gen_beta2", 0.9))
        
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_config.get("disc_lr", 1e-5),
                                            beta_1=disc_config.get("disc_beta1", 0.5),
                                            beta_2=disc_config.get("disc_beta2", 0.9))

        self.dataset = dataset
        self.train_ind = 0
        self.trained_data = []

        self.loss = {'disc': [], 'gen': []}

        self.load_history() # Load any history we can find in save directory

        atexit(self.save_model)

    def prepare_batch(self, datatype, batch_size):
        count = 0
        batch = []

        if datatype not in ['magnified_profile', 'profile', 'core']:
            print(f"ERROR: The datatype {datatype} passed in is an available datatype (i.e. 'magnified_profile', 'profile', 'core')")
            return None
        
        while count < batch_size and self.train_ind < len(self.dataset['train']):
            sample = self.dataset['train'][self.train_ind]

            if sample['datatype'] == datatype:

                image = sample['image']

                if not isinstance(image, tf.Tensor):
                    image = tf.convert_to_tensor(np.array(image))  # Convert from PIL to tensor
                
                image = tf.image.resize(image, self.gen_config.get('resolution', (512, 512)))
                
                image = tf.cast(image, tf.float32) / 127.5 - 1.0
                
                batch.append(image)

                count += 1

            self.train_ind += 1

            if self.train_ind >= len(self.dataset['train']):
                self.train_ind = 0

        if len(batch) > 0:
            return np.stack(batch)
        
        else:
            print("All images have been assessed and no available images could be found")
            return None


    def train(self, batches = None, batch_size = 8, epochs = None):
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
        if not batches: batches = int(round(len(self.dataset['train'])/batch_size, 0))
        if batch_size: self.batch_size = batch_size
        if epochs: self.epochs = epochs

        trainable_data = True

        # Iterate through requested training batches 
        while trainable_data:
            # Figure out batch start and end

            x = self.prepare_batch('magnified_profile', batch_size) # Load a new batch of subjects

            print(f"Training on batch {batch + 1}...")
            for epoch in range(self.epochs): # Iterate through each requested epoch
                self.train_step(x) # Train on batch of images
                print(f'Epoch {epoch + 1} | Generator loss: {round(float(self.loss['gen'][-1]), 3)} | Discrimintator loss: {round(float(self.loss['disc'][-1]), 3)} |')
                
                self.plot_history() # Update history with progress
            
            # Generate synthetic images to the batch folder to track progress
            if self.synthetics:
                _ = generate(self.gen, count = self.synthetics, seed_size = self.gen_config.get('latent_dim', 100), save_dir = f"{self.save_dir}/synthetic_images/", filename_prefix = f'batch_{batch}_synthetic')
            
            # Save the models state
            if batch % 10 == 0:
                self.save_model(f"{self.save_dir}/synthetic_images/batch_{batch}") # Need to consider more dynamic way to do this and remove old history

    def train_step(self, images, disc_steps = 1, gen_steps = 3):
        """
        This function trains the snowGAN on the passed in images, with variables 
        training length of the discriminator and generator. This function implements
        WGAN-GP unique EMD loss and gradient penalty regularization.
        
        Function arguments:
            images (4D numpy array) - numpy array of RGB images set to the class resolution (sample axis = 0)
            disc_steps (int) - Number N of training steps for the discriminator per epoch
            gen_steps (int) - Number M of training steps for the generator per epoch
        """
        # Set batch size to the number of images passed in
        batch_size = tf.shape(images)[0] 
        # Generate noise for generator input
        noise = tf.random.normal([batch_size, self.noise_dimension]) 
        # Train the discriminator N times
        for _ in range(disc_steps): 
            with tf.GradientTape() as tape: # Initialize automatic differentiation during forward propogation
                # Generate a synthetic sample via forward propogation in the generator
                synthetic_images = self.gen.model(noise, training=True) 

                # Forward propogate real & synethic images to the discriminator
                output = self.disc.model(images, training=True) 
                synthetic_output = self.disc.model(synthetic_images, training=True)

                # Calculate discriminators gradient penalty
                gp = self.disc.compute_gp(images, synthetic_images)

                # Calculate EMD/loss for the discriminators outputs
                disc_loss = self.disc.loss(output, synthetic_output, gp, self.lambda_gp)
            
            # Backpropogate by calculating gradient 
            disc_gradients = tape.gradient(disc_loss, self.disc.model.trainable_variables)
            # Apply gradients via the optimizer and back propogation
            self.disc.optimizer.apply_gradients(zip(disc_gradients, self.disc.model.trainable_variables))

        # Train the generator M times
        for _ in range(gen_steps):
            # Generate random noise to pass into the generator
            noise = tf.random.normal([batch_size, self.noise_dimension])
            # Initialize automtic differentiation during forward propogation
            with tf.GradientTape() as tape:
                # Forward pass noise through the generator to create synthetic images
                synthetic_images = self.gen.model(noise, training=True)
                # Forward pass generator outputs through the discriminator for loss calculation
                synthetic_output = self.disc.model(synthetic_images, training=True)
                # Calculate generator loss for backpropogation by calculating mean of disc output
                gen_loss = self.gen.loss(synthetic_output)
            # Backpropogate to calculate gradient of trainable parameters given loss
            gen_gradients = tape.gradient(gen_loss, self.gen.model.trainable_variables)
            # Apply the gradients and update trainable parameters (w, b, etc.)
            self.gen.optimizer.apply_gradients(zip(gen_gradients, self.gen.model.trainable_variables))

        # Record loss of models to their histories - Should this be within training loops? Might be deceptive if not
        self.loss['gen'].append(gen_loss)
        self.loss['disc'].append(disc_loss)

    def save_model(self, path = None):
        """
        Save the currently loaded generator and discriminator to a given path

        Function arguments:
            path (str) - Folder to save the generator and discriminator .keras files
        """
        # Update path if none passed in
        if path == None:
            path = self.save_dir

        # Log and plot final history
        self.log_history()
        self.plot_history()
        self.log_training()
        
        # Save generator and discriminator as .keras files
        self.model.gen.model.save(f"{path}/generator.keras")
        self.model.disc.model.save(f"{path}/discriminator.keras")
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

    def log_history(self):
        """
        Plot and save the generator and discriminator history loaded in the snowGAN object
        """

        # Save the current generate loss progress
        with open(f"{self.save_dir}generator_loss.txt", "w") as file:
            for loss in self.loss['gen']:
                file.write(f"{loss}\n")

        # Save the current discriminator loss
        with open(f"{self.save_dir}discriminator_loss.txt", "w") as file:
            for loss in self.loss['disc']:
                file.write(f"{loss}\n")
    
        with open(f"{self.save_dir}trained.txt", "w") as file:
            for trained in self.trained_data:
                file.write(f"{trained}\n")

    def load_history(self):
        """
        Load generator/discriminator loss history and trained data from text files
        and assign them to the snowGAN object.
        """
        import os

        gen_path = os.path.join(self.save_dir, "generator_loss.txt")
        disc_path = os.path.join(self.save_dir, "discriminator_loss.txt")
        trained_path = os.path.join(self.save_dir, "trained.txt")

        # Initialize containers
        self.loss = {"gen": [], "disc": []}
        self.trained_data = []

        # Load generator loss
        if os.path.exists(gen_path):
            with open(gen_path, "r") as file:
                self.loss["gen"] = [float(line.strip()) for line in file if line.strip()]
        
        # Load discriminator loss
        if os.path.exists(disc_path):
            with open(disc_path, "r") as file:
                self.loss["disc"] = [float(line.strip()) for line in file if line.strip()]

        # Load trained data
        if os.path.exists(trained_path):
            with open(trained_path, "r") as file:
                self.trained_data = [line.strip() for line in file if line.strip()]



    def plot_history(self):
        # Plot the generator and discriminator loss history
        plt.plot(self.loss['gen'], label = 'Generator loss')
        plt.plot(self.loss['disc'], label = 'Discriminator loss')
        plt.title("GAN History")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.savefig(f'{self.save_dir}history.png')
        plt.close()


