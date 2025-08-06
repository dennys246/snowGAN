import tensorflow as tf

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

        # Check if default config, and reverse if so
        if self.config.filter_counts[0] < self.config.filter_counts[-1]:
            self.config.filter_counts = self.config.filter_counts[::-1]

        self.model = self._build_model()

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.config.latent_dim,))

        x = tf.keras.layers.Dense(16 * 16 * self.config.resolution[0], use_bias = False)(inputs)
        x = tf.keras.layers.Reshape((16, 16, self.config.resolution[0]))(x)

        for filters in self.config.filter_counts:
            x = tf.keras.layers.Conv2DTranspose(filters, self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.padding)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)

        outputs = tf.keras.layers.Conv2DTranspose(3, self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.padding, activation = self.config.final_activation, use_bias = False)(x)

        return tf.keras.Model(inputs, outputs, name = "Generator")

    def get_optimizer(self, lr = 1e-4, beta_1 = 0.5, beta_2 = 0.9):
        return tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = beta_1, beta_2 = beta_2)

