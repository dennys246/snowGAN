import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self, config):
        """
        Wasserstein GAN Discriminator: distinguishes real vs. synthetic snow images.
        
        Args:
            config
        """
        super(Discriminator, self).__init__()
        self.config = config

        self.resolution = self.config.resolution
        self.model = self._build_model()

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.config.resolution[0], self.config.resolution[1], 3))
        x = inputs

        for filters in self.config.filter_counts:
            x = tf.keras.layers.Conv2D(filters, self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.padding)(x)
            x = tf.keras.layers.LeakyReLU(alpha = self.config.alpha)(x)

        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(1)(x)  # No activation for WGAN

        return tf.keras.Model(inputs, outputs, name="Discriminator")

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate = config.learning_rate, beta_1 = config.beta_1, beta_2 = config.beta_2)

