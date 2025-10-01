import os
import tensorflow as tf

from snowgan.config import load_gen_config

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
        self.optimizer = self.get_optimizer()

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.config.latent_dim,))

        x = tf.keras.layers.Dense(16 * 16 * self.config.filter_counts[0], use_bias = False)(inputs)
        x = tf.keras.layers.Reshape((16, 16, self.config.filter_counts[0]))(x)

        for filters in self.config.filter_counts:
            x = tf.keras.layers.Conv2DTranspose(filters, self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.padding)(x)
            #x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(self.config.negative_slope)(x)

        outputs = tf.keras.layers.Conv2DTranspose(3, self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.padding, activation = self.config.final_activation, use_bias = False)(x)

        return tf.keras.Model(inputs, outputs, name = "Generator")

    def get_optimizer(self, lr = 1e-4, beta_1 = 0.5, beta_2 = 0.9):
        return tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = beta_1, beta_2 = beta_2)

    def get_loss(self, synthetic_difference):
        """
        Generator loss for Wasserstein GAN loss

        Args:
            synthetic_output (tf.Tensor): Discriminator output for fake images

        Returns:
            tf.Tensor: Generator loss
        """
        return -tf.reduce_mean(synthetic_difference)

def load_generator(model_path):
    split = model_path.split("/")

    # Check if the model path exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}, creating a rebuild generator model.")
        os.makedirs("/".join(split[:-1]), exist_ok = True)

    config = load_gen_config("/".join(split[:-1]) + "/generator.keras")
    
    config.checkpoint = split.pop() # Get the model filename from the path
    config.save_dir = "/".join(split) + "/"

    # Load the discriminator
    generator = Generator(config)
    generator.model.build((config.resolution[0], config.resolution[1], 3))
    return generator