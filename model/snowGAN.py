import os, atexit, argparse, shutil, re, cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Configure tensorflow
from tensorflow.keras.mixed_precision import set_global_policy
tf.config.optimizer.set_jit(True)  # Use XLA computation for faster runtime operations

class snowGAN:

    def __init__(self, path = "./", resolution = (1024, 1024), new = False):
        """
        The abominable snowGAN is a generative adversarial network (GAN) used to train and 
        generate synthetic samples of magnified images of snowpack sampled through out the
        Rocky Mountains. The specific architecture implemented is Wasserstein GAN with gradient
        penalty (WGAN-GP)leveraging earth movers distance (EMD) as a loss function for the
        disriminator and gradient penalty as a regularization to manage vanishing and exploding
        gradients common to convolutional neural networks (CNN) and GAN's.
        

        Attributes:
            path (str) - Path to save the models or containing a pre-trained model to load
            resolution (set of int) - Resolution to resample snow images too
            new (bool) - Boolean whether to reinitialize models weights before training
        
        Methods:
            train() - Initializes a training session for iterating through a set of images
            learn() -  Orchestrates training steps for both the discriminator and generator
            generate() - Generates a synthetic image using the currently loaded generator
            save_model() - Saves the generator & discriminator models, histories and training progress
            load_model() - Loads a pre-trained generator & discriminator models, histories and training progress
            plot_history() - Plots the current training history
            make_movie() - Creates a .mp4 movie of images generates over learning batches to showcase learning progress
        """

        # Check if model folder rebuilt requested
        if new and os.path.exists(path):
            # Double check with user model is to be rebuilt
            response = input("New model requested, are you sure you would like to delete the current model? (y/n)\n")
            if response == 'y': # Rebuild
                shutil.rmtree(path)
            else: # Cancel rebuild
                print("Model rebuilding canceled, reverting to loading old pre-trained model")
                new = False

        self.path = path # Handle the model path
        if not os.path.exists(self.path):
            os.makedirs(f"{self.path}synthetic_images/", exist_ok = True)

        # Initialize generator and discriminator
        self.disc = _discriminator(self.resolution)
        self.gen = _generator(self.resolution)

        # Check if model could be loaded
        if path and new == False:
            if os.path.exists(f"{self.path}generator.keras"):
                self.load_model()

        # Define training runtime parameters
        self.resolution = resolution # Resolution to resample images to
        
        self.noise_dimension = 100 # Length of noise seed to be fed into generator
        self.lambda_gp = 10.0 # Regularization parameter

        atexit.register(self.save_model)


    def learn(self, images, disc_steps = 1, gen_steps = 3):
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


class _discriminator:

    def __init__(self, resolution):
        """
        This class implements a Wasserstein GAN discriminator convolution neural network
        along with earth mover distance (EMD) loss and gradient penalty. The discriminator
        activity is largely mediated through the snowGAN main class.

        Class attributes:
            resolution (set of int) - The resolution of the images being passed into the model

        Class functions:
            build() - Builds the discriminator convolutional neural network architecture
            loss() - Calculates the loss of the discriminator using Wasserstein EMD loss and gradient penalty
            compute_gp() - Computes gradient penalty for WGAN-GP and the learning function
        """
        self.resolution = resolution
        self.build()

    def build(self, lr = 1e-5, b_1 = 0.5, b_2 = 0.9):
        """
        Builds a classic discriminator architecture for the snowGAN which inputs either a real or synthetic image
        and then passes the information through consecutive layers of convolutional and leaky relu till it reaches a
        final layer outputting a binary classification output real/synthetic.
        
        Function arguments:
            lr (float) - Learning rate for the 
            b_1 (float) - Beta 1 hyperparameter for optimizer
            b_2 (float) - Beta 2 hyperparameter for optimizer
        """
        inputs = tf.keras.Input(shape=(self.resolution[0], self.resolution[1], 3))

        # First block
        x = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU(0.25)(x)

        # Second block
        x = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.25)(x)

        # Third block
        x = tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.25)(x)

        # Fourth block
        x = tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.25)(x)

        # Fifth block
        x = tf.keras.layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.25)(x)

        # Output
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(1)(x)  # Binary classification (real/synthetic)

        # Build the model
        self.model = tf.keras.Model(inputs, outputs, name="Discriminator")
        self.model.summary()

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b_1, beta_2=b_2)

    def loss(self, real_output, synthetic_output, gp, lambda_gp = 10.0):
        """
        Calculate earth mover distance (EMD) or Wasserstein's loss using gradient penalty

        Function arguments:
            real_output (2D numpy array) - Discriminator output from real images inputted
            synthetic_output (2D numpy array) - Discriminator output from syntethic images inputted
            gp (float) - Gradient penalty calculated fomr the batch of real and synthetic images
            lambda_gp (float) - Regularization term for scaling gradient penalty (default 10.0)
        """
        # Calculate loss
        wasserstein_loss = tf.reduce_mean(synthetic_output) - tf.reduce_mean(real_output)
        # Apply gradient penalty and return
        return wasserstein_loss + lambda_gp * gp  

    def compute_gp(self, images, synthetic_images):
        """
        Calculates a gradient penalty for regularization in Wasserstein GANs utilizing a
        a 1-Lipschitz constraint to penalize large gradients largely by smoothing the gradient.

        Function arguments:
            images (4D numpy array) - numpy array of RGB images (sample axis = 0)
            synthetic images (4D numpy array) - numpy array of synthetic RGB images (sample axis = 0)
        """
        batch_size = tf.shape(images)[0]
        # Generate a random interpolation factor for each image in the batch
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)  
        # Interpolate images between a real and fake images using the interpolation factors
        interpolated = alpha * images + (1 - alpha) * synthetic_images  

        # Forward pass interpolated images through the discriminator
        with tf.GradientTape() as tape: 
            tape.watch(interpolated) # Ask to track interpolated gratiends
            pred = self.disc.model(interpolated, training=True)

        # Calculate the gradients of discriminator via backpropogration
        grads = tape.gradient(pred, [interpolated])[0]
        
        # Calculate L2 norm
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))

         # Final computation for gradient penalty regularization term
        penalty = tf.reduce_mean((norm - 1.0) ** 2) * self.lambda_gp
        return penalty

class _generator:

    def __init__(self, resolution):
        """
        This class implements a Wasserstein GAN generator that inputs noise and outputs a 
        synthetic image of snow. The architecture implemented includes series of convolutional
        tranpositional layers paired with batch normalization and leaky reLU activation. Activity 
        is largely mediated through the snowGAN main class.

        Class attributes:
            resolution (set of int) - The resolution of the images being passed into the model

        Class functions:
            build() - Builds the discriminator convolutional neural network architecture
            loss() - Calculates the loss of the discriminator using Wasserstein EMD loss and gradient penalty
        """
        # Set class variables
        self.resolution = resolution

        self.build() # Call for the tensorflow model to build

    def build(self, lr = 1e-4, b_1 = 0.5, b_2 = 0.9):
        """
        Builds a classic generator architecture for the snowGAN which inputs noise and then 
        passes the noise through consecutive layers of convolutional transpositional layers
        till it reaches a final resolution
        
        Function arguments:
            lr (float) - Learning rate for the 
            b_1 (float) - Beta 1 hyperparameter for optimizer
            b_2 (float) - Beta 2 hyperparameter for optimizer
        """
        latent_dim = 100  # Assuming your input latent vector size

        inputs = tf.keras.Input(shape=(latent_dim,))

        # Dense projection + reshape
        x = tf.keras.layers.Dense(16 * 16 * 1024, use_bias=False)(inputs)
        x = tf.keras.layers.Reshape((16, 16, 1024))(x)

        # Block 1
        x = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        # Block 2
        x = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        # Block 3
        x = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        # Block 4
        x = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        # Block 5
        x = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        # Output layer
        outputs = tf.keras.layers.Conv2DTranspose(
            3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'
        )(x)

        # Build model
        self.model = tf.keras.Model(inputs, outputs, name="Generator")
        self.model.summary()

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b_1, beta_2=b_2)

    def loss(self, synthetic_output):
        """
        Calculates the loss for the generator which is simpyl the mean of the discriminator output 
        from synthetically generated images being inputted. Basically calculating accuracy of the
        discriminator, and ultimately pushing the generator to minimize discriminator accuracy.

        Function arguments:
            synthetic_output (2D numpy array) - Output from the discriminator for snythetic images
        """
        return -tf.reduce_mean(synthetic_output) # Calculate mean of discriminators output on generated images



if __name__ == "__main__": # Add in command line functionality

    # Initialize the parser for accepting arugments into a command line call
    parser = argparse.ArgumentParser(description = "The snowGAN model is used to train a GAN on a dataset of snow samples magnified on a crystal card. You can define how the model runs by the number of epochs, batch sizes and other parameters. You can also pass in a path to a pre-trained snowGAN to accomplish transfer learning on new GAN tasks!")

    # Add command-line arguments
    parser.add_argument('--path', type = str, default = "snowgan/", help = "Path to a pre-trained model or directory to save results (defaults to snowgan/)")
    parser.add_argument('--resolution', type = set, default = (1024, 1024), help = 'Resolution to downsample images too (Default set to (1024, 1024))')
    parser.add_argument('--batches', type = int, default = None, help = 'Number of batches to run (Default to max available)')
    parser.add_argument('--batch-size', type = int, default = 8, help = 'Batch size (Defaults to 8)')
    parser.add_argument('--epochs', type = int, default = 100, help = 'Training epochs per image (Defaults to 100)')
    parser.add_argument('--new', type = bool, default = False, help = 'Whether to rebuild model from scratch (defaults to False)')

    # Parse the arguments
    args = parser.parse_args()

    # Create the snowGAN object with the parsed arguments
    snowgan = snowGAN(path = args.path, resolution = args.resolution, new = args.new)

    # Train the model
    snowgan.train(batches = args.batches, batch_size = args.batch_size, epochs = args.epochs)

    # Generate a final batch of images for viewing
    snowgan.generate()
