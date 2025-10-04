import os
import tensorflow as tf

from snowgan.config import build

class Generator(tf.keras.Model):
    def __init__(self, config):
        """
        Wasserstein GAN Generator that converts latent vectors into synthetic snow images.
        
        Args:
            resolution (tuple): Target image resolution, e.g. (512, 512)
            latent_dim (int): Dimension of the input latent vector
        """
        
        super(Generator, self).__init__()
        self.config = config

        # Check if first is smaller than last filter (i.e. discriminator setup) switch
        if self.config.filter_counts[0] < self.config.filter_counts[-1]:
            print(f"Inverse filter counts detected, inverting filter counts - {self.config.filter_counts}")
            self.config.filter_counts = self.config.filter_counts[::-1]

        self.model = self._build_model()
        self.optimizer = self.get_optimizer()

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.config.latent_dim,))

        x = tf.keras.layers.Dense(16 * 16 * self.config.filter_counts[0], use_bias = False)(inputs)
        x = tf.keras.layers.Reshape((16, 16, self.config.filter_counts[0]))(x)

        for filters in self.config.filter_counts:
            x = tf.keras.layers.Conv2DTranspose(filters, self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.padding)(x)
            if self.config.batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(self.config.negative_slope)(x)

        outputs = tf.keras.layers.Conv2DTranspose(3, self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.padding, activation = self.config.final_activation, use_bias = False)(x)

        return tf.keras.Model(inputs, outputs, name = "Generator")

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate = self.config.learning_rate, beta_1 = self.config.beta_1, beta_2 = self.config.beta_2)

    def get_loss(self, synthetic_difference):
        """
        Generator loss for Wasserstein GAN loss

        Args:
            synthetic_output (tf.Tensor): Discriminator output for fake images

        Returns:
            tf.Tensor: Generator loss
        """
        return -tf.reduce_mean(synthetic_difference)

def load_generator(checkpoint, config = None):
    
    if not config:
        split = checkpoint.split("/")
        config = build("/".join(split[:-1]) + "/generator.keras")
    
        # Get the model filename from the path
        config.checkpoint = checkpoint
        config.save_dir = "/".join(split) + "/"

    # Load the discriminator
    generator = Generator(config)
    generator.model.build((config.resolution[0], config.resolution[1], 3))
    return generator