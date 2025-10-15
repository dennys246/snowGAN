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
            print(f"Inverse filter counts detected, inverting filter counts (Normal if using default config for generator) - {self.config.filter_counts}")
            self.config.filter_counts = self.config.filter_counts[::-1]

        self.model, self.fade_endpoints = self._build_model()
        self.optimizer = self.get_optimizer()

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.config.latent_dim,))

        x = tf.keras.layers.Dense(16 * 16 * self.config.filter_counts[0], use_bias=False)(inputs)
        x = tf.keras.layers.Reshape((16, 16, self.config.filter_counts[0]))(x)

        feats = []
        for filters in self.config.filter_counts:
            x = tf.keras.layers.Conv2DTranspose(filters, self.config.kernel_size, strides=self.config.kernel_stride, padding=self.config.padding)(x)
            if self.config.batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(self.config.negative_slope)(x)
            feats.append(x)

        curr_img = tf.keras.layers.Conv2DTranspose(3, self.config.kernel_size, strides=self.config.kernel_stride, padding=self.config.padding, activation=self.config.final_activation, use_bias=False, name="toRGB_curr")(x)

        base_model = tf.keras.Model(inputs, curr_img, name="Generator")

        # Build fade endpoints model (prev_up_rgb, curr_rgb) if we have >= 2 blocks
        fade_model = None
        if len(feats) >= 2:
            prev_feat = feats[-2]
            # Use bias so the prev path can avoid collapsing to near-zero early
            prev_rgb = tf.keras.layers.Conv2D(3, kernel_size=1, padding='same', activation=self.config.final_activation, use_bias=True, name="toRGB_prev")(prev_feat)
            # Dynamically resize previous RGB to match current output spatial dims
            prev_up = tf.keras.layers.Lambda(
                lambda xs: tf.image.resize(xs[0], size=tf.shape(xs[1])[1:3], method='nearest'),
                name="prev_resize_to_curr"
            )([prev_rgb, curr_img])

            fade_model = tf.keras.Model(inputs, [prev_up, curr_img], name="GeneratorFadeEndpoints")

        return base_model, fade_model

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
        
        config.checkpoint = checkpoint # Get the model filename from the path
        config.save_dir = "/".join(split) + "/"


    # Load the discriminator
    generator = Generator(config)
    generator.model.build((config.resolution[0], config.resolution[1], 3))
    # Build fade endpoints as well if present (latent input shape)
    if generator.fade_endpoints is not None:
        generator.fade_endpoints.build((None, config.latent_dim))
    return generator
