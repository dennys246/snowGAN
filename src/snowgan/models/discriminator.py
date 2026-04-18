import os
import tensorflow as tf
import keras

from snowgan.config import build

class Discriminator(keras.Model):
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
        inputs = keras.Input(shape=(self.config.depth, self.config.resolution[0], self.config.resolution[1], self.config.channels))
        x = inputs
        use_sn = getattr(self.config, 'spectral_norm', False)

        ksize = (1, self.config.kernel_size[0], self.config.kernel_size[1])
        kstride = (1, self.config.kernel_stride[0], self.config.kernel_stride[1])

        for filters in self.config.filter_counts:
            conv = keras.layers.Conv3D(filters, ksize, strides=kstride, padding=self.config.padding)
            x = keras.layers.SpectralNormalization(conv)(x) if use_sn else conv(x)
            x = keras.layers.LeakyReLU(negative_slope=self.config.negative_slope)(x)


        x = keras.layers.Flatten()(x)
        dense = keras.layers.Dense(1)  # No activation for WGAN
        outputs = keras.layers.SpectralNormalization(dense)(x) if use_sn else dense(x)

        return keras.Model(inputs, outputs, name="Discriminator")

    def get_optimizer(self):
        return keras.optimizers.Adam(learning_rate = self.config.learning_rate, beta_1 = self.config.beta_1, beta_2 = self.config.beta_2)

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
        # Cast to float32 for loss computation (mixed precision outputs float16)
        wasserstein = tf.reduce_mean(tf.cast(synthetic_output, tf.float32)) - tf.reduce_mean(tf.cast(real_output, tf.float32))
        return wasserstein + lambda_gp * tf.cast(gp, tf.float32)
    
def load_discriminator(checkpoint, config = None):
    
    if not config:
        split = checkpoint.split("/")
        config = build("/".join(split[:-1]) + "/discriminator.keras")
        config.checkpoint = split.pop() # Get the model filename from the path
        config.save_dir = "/".join(split) + "/"

    # Load the discriminator
    discriminator = Discriminator(config)
    discriminator.model.build((None, config.depth, config.resolution[0], config.resolution[1], config.channels))
    return discriminator
