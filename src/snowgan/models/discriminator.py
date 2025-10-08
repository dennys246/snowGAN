import os
import tensorflow as tf

from snowgan.config import build

class Discriminator(tf.keras.Model):
    def __init__(self, config):
        """
        Wasserstein GAN Discriminator: distinguishes real vs. synthetic snow images.
        
        Args:
            config
        """
        super(Discriminator, self).__init__()
        self.config = config

        # Check if first is larger than last filter (i.e. discriminator setup) switch
        if self.config.filter_counts[0] > self.config.filter_counts[-1]:
            print(f"WARNING: Inverse filter counts detected, inverting filter counts - {self.config.filter_counts}")
            self.config.filter_counts = self.config.filter_counts[::-1]

        self.resolution = self.config.resolution
        self.model = self._build_model()
        self.optimizer = self.get_optimizer()

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.config.resolution[0], self.config.resolution[1], 3))
        x = inputs

        for filters in self.config.filter_counts:
            x = tf.keras.layers.Conv2D(filters, self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.padding)(x)
            x = tf.keras.layers.LeakyReLU(negative_slope = self.config.negative_slope)(x)

        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(1)(x)  # No activation for WGAN

        return tf.keras.Model(inputs, outputs, name="Discriminator")

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate = self.config.learning_rate, beta_1 = self.config.beta_1, beta_2 = self.config.beta_2)

    def get_loss(self, real_output, synthetic_output, gp, lambda_gp):
        """
        Wasserstein loss with gradient penalty for the Discriminator.

        Args:
            real_output (tf.Tensor): Discriminator output for real images
            synthetic_output (tf.Tensor): Discriminator output for fake images
            gp (tf.Tensor): Gradient penalty term
            lambda_gp (float): Weight for gradient penalty

        Returns:
            tf.Tensor: Total discriminator loss
        """
        wasserstein = tf.reduce_mean(synthetic_output) - tf.reduce_mean(real_output)
        return wasserstein + lambda_gp * gp
    
def load_discriminator(checkpoint, config = None):
    
    if not config:
        split = checkpoint.split("/")
        config = build("/".join(split[:-1]) + "/discriminator.keras")

<<<<<<< Updated upstream
        config.checkpoint = split.pop() # Get the model filename from the path
        config.save_dir = "/".join(split) + "/"
=======
    # Check if the model path exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}, creating a rebuild discriminator model.")
        os.makedirs("/".join(split[:-1]), exist_ok = True)

    config = load_disc_config("/".join(split[:-1]) + "/discriminator.keras")

    config.checkpoint = split.pop() # Get the model filename from the path
    config.save_dir = "/".join(split) + "/"
>>>>>>> Stashed changes

    # Load the discriminator
    discriminator = Discriminator(config)
    discriminator.model.build((config.resolution[0], config.resolution[1], 3))
    return discriminator