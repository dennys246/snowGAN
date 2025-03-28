import os, sys, atexit, argparse, shutil, re, cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pipeline import pipeline

from tensorflow.keras.mixed_precision import set_global_policy

class snowGAN:
    def __init__(self, path = "snowgan/", resolution = (1024, 1024), new = False):
        if new and os.path.exists(path):
            response = input("New model requested, are you sure you would like to delete the current model? (y/n)\n")
            if response == 'y':
                shutil.rmtree(path)
            else:
                print("Model rebuilding canceled, reverting to loading old pre-trained model")
                new = False

        self.path = path # Handle the model path
        if not os.path.exists(self.path):
            os.makedirs(f"{self.path}synthetic_images/", exist_ok = True)

        self.resolution = resolution

        self.pipe = pipeline(self.path, self.resolution)
        self.disc = discriminator(self.resolution)
        self.gen = generator(self.resolution)
        self.loss = {'disc': [], 'gen': []}

        if path and new == False:
            if os.path.exists(f"{self.path}generator.keras"):
                self.load_model()

        self.batch_size = 8
        self.epochs = 50
        self.noise_dimension = 100
        self.synthetics = 25
        self.lambda_gp = 10.0

        atexit.register(self.save_model)

    def learn_wgan_gp(self, images, disc_steps = 1, gen_steps = 3):
            batch_size = tf.shape(images)[0]
            noise = tf.random.normal([batch_size, self.noise_dimension])

            for _ in range(disc_steps):  # Train Discriminator multiple times
                with tf.GradientTape() as tape:
                    synthetic_images = self.gen.model(noise, training=True)
                    output = self.disc.model(images, training=True)
                    synthetic_output = self.disc.model(synthetic_images, training=True)
                    gp = self.gradient_penalty(images, synthetic_images)
                    disc_loss = self.disc.loss(output, synthetic_output, gp, self.lambda_gp)

                disc_gradients = tape.gradient(disc_loss, self.disc.model.trainable_variables)
                self.disc.optimizer.apply_gradients(zip(disc_gradients, self.disc.model.trainable_variables))

            # Train Generator
            for _ in range(gen_steps):
                noise = tf.random.normal([batch_size, self.noise_dimension])
                with tf.GradientTape() as tape:
                    synthetic_images = self.gen.model(noise, training=True)
                    synthetic_output = self.disc.model(synthetic_images, training=True)
                    gen_loss = self.gen.loss(synthetic_output)

                gen_gradients = tape.gradient(gen_loss, self.gen.model.trainable_variables)
                self.gen.optimizer.apply_gradients(zip(gen_gradients, self.gen.model.trainable_variables))

            self.loss['gen'].append(gen_loss)
            self.loss['disc'].append(disc_loss)

    def learn(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dimension])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.gen.model(noise, training=True)

            real_output = self.disc.model(images, training=True)
            synthetic_output = self.disc.model(generated_images, training=True)

            gen_loss = self.gen.loss(synthetic_output)
            self.loss['gen'].append(gen_loss.numpy())

            disc_loss = self.disc.loss(real_output, synthetic_output, self.lambda_gp)
            self.loss['disc'].append(disc_loss.numpy())

        gen_grad = gen_tape.gradient(gen_loss, self.gen.model.trainable_variables)
        gen_grad, gen_variables = zip(*self.gen.optimizer.compute_gradients(gen_loss, self.gen.model.trainable_variables))
        self.gen.optimizer.apply_gradients(zip(gen_grad, gen_variables))

        disc_grad = disc_tape.gradient(disc_loss, self.disc.model.trainable_variables)
        disc_grad, disc_variables = zip(*self.disc.optimizer.compute_gradients(disc_loss, self.disc.model.trainable_variables))
        self.disc.optimizer.apply_gradients(zip(disc_grad, disc_variables))

    def train(self, batches = None, batch_size = 8, epochs = None):
        # Update hyperparameters if passed in before training
        if not batches:
            batches = int(round(len(self.pipe.avail_photos)/batch_size, 0))
        if batch_size: self.batch_size = batch_size
        if epochs: self.epochs = epochs

        previous_batches = len(glob(f"{self.path}/synthetic_images/batch_*/"))
        batches += previous_batches

        for batch in range(previous_batches, batches):
            self.x = self.pipe.load_batch(self.batch_size)

            print(f"Training on batch {batch + 1}...")
            for epoch in range(self.epochs):
                self.learn_wgan_gp(self.x)
                print(f'Epoch {epoch + 1} | Generator loss: {round(float(self.loss['gen'][-1]), 3)} | Discrimintator loss: {round(float(self.loss['disc'][-1]), 3)} |')
                self.plot_history()

            _ = self.generate(f"synthetic_batch_{batch}", f"Synthetic Image after {round((float(batch)/float(batches))*100, 2)}%", f"batch_{batch}/")
            self.save_model(f"synthetic_images/batch_{batch}")

    def gradient_penalty(self, images, synthetic_images):
        batch_size = tf.shape(images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)  # Random weight
        interpolated = alpha * images + (1 - alpha) * synthetic_images  # Mix images

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.disc.model(interpolated, training=True)

        grads = tape.gradient(pred, [interpolated])[0]  # Compute gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))  # L2 norm
        penalty = tf.reduce_mean((norm - 1.0) ** 2)  # Gradient penalty
        return penalty

    def generate(self, filename_prefix = 'synthetic', title = "Synthetic Image", subfolder = ""):
        seed = tf.random.normal([self.synthetics, self.noise_dimension])  # Fixed seed for consistent visualizatio
        synthetic_images = self.gen.model(seed, training=False)
        print(f"Synthetic images generated: {synthetic_images.shape}")
        for ind in range(synthetic_images.shape[0]):
            # Make sure folders exist
            if subfolder:
                os.makedirs(f"{self.path}synthetic_images/{subfolder}", exist_ok = True)
            filename = f"{self.path}synthetic_images/{subfolder}{filename_prefix}_image_{ind}.png"

            # Reformat generated image to RGB
            image = synthetic_images[ind].numpy()
            print(f"Image (prior) shape {image.shape} | max {image.max()} | deviation {image.std()}")
            image = np.clip((image + 1) * 127.5, 0, 255).astype(np.uint8)
            print(f"Image (prior) shape {image.shape} | max {image.max()} | deviation {image.std()}")

            # Save image with parameters provided
            plt.imshow(image)
            plt.title(title)
            plt.axis('off')
            plt.savefig(filename)
            plt.close()
        return synthetic_images

    def save_model(self, path_extension = None):
        path = self.path
        if path_extension:
            path += path_extension
        self.gen.model.save(f"{path}/generator.keras")
        self.disc.model.save(f"{path}/discriminator.keras")
        print(f"Models saved in {path}...")

    def load_model(self, path_extension = None):
        path = self.path
        if path_extension:
            path += path_extension
        self.gen.model = tf.keras.models.load_model(f"{path}/generator.keras")
        self.disc.model = tf.keras.models.load_model(f"{path}/discriminator.keras")
        print(f"Models loaded from {path}...")

    def plot_history(self):
        plt.plot(self.loss['gen'], label = 'Generator loss')
        plt.plot(self.loss['disc'], label = 'Discriminator loss')
        plt.title("GAN History")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.savefig(f'{self.path}history.png')
        plt.close()


        with open(f"{self.path}generator_loss.txt", "w") as file:
            for loss in self.loss['gen']:
                file.write(f"{loss}\n")

        with open(f"{self.path}discriminator_loss.txt", "w") as file:
            for loss in self.loss['disc']:
                file.write(f"{loss}\n")

    def make_movie(self, videoname = "snowgan.mp4", framerate = 15):
        videoname = f"{self.path}/{videoname}"
        synthetic_files = sorted(glob(f"{self.path}/synthetic_images/batch_*/*.png"))
        pattern = re.compile(r"batch_(\d+)/synthetic_batch_\d+_image_(\d+)\.png")
        batches = {}
        for file in synthetic_files:
            match = pattern.search(file)
            batch = int(match.group(1))
            if batch not in batches.keys():
                batches[batch] = []
            batches[batch].append(file)
        
        # Read the first image to get dimensions
        image = cv2.imread(batches[0][0])
        height, width, layers = image.shape

        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
        video = cv2.VideoWriter(videoname, fourcc, framerate, (width, height))

        # Add images to the video
        for batch in sorted(batches.keys()):
            for image_file in batches[batch]:
                image = cv2.imread(image_file)
                video.write(image)

        # Release the video writer
        video.release()
        cv2.destroyAllWindows()


class discriminator:

    def __init__(self, resolution):
        self.resolution = resolution
        self.build()

    def build(self, lr = 1e-5):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same', input_shape = (self.resolution[0], self.resolution[1], 3)))
        self.model.add(tf.keras.layers.LeakyReLU(0.25))

        self.model.add(tf.keras.layers.Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
        self.model.add(tf.keras.layers.LeakyReLU(0.25))

        self.model.add(tf.keras.layers.Conv2D(256, (5, 5), strides = (2, 2), padding = 'same'))
        self.model.add(tf.keras.layers.LeakyReLU(0.25))

        self.model.add(tf.keras.layers.Conv2D(512, (5, 5), strides = (2, 2), padding = 'same'))
        self.model.add(tf.keras.layers.LeakyReLU(0.25))

        self.model.add(tf.keras.layers.Conv2D(1024, (5, 5), strides = (2, 2), padding = 'same'))
        self.model.add(tf.keras.layers.LeakyReLU(0.25))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1))  # Binary classification (real/synthetic)

        self.model.build()
        self.model.summary()

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.5, beta_2 = 0.9)

    def loss(self, real_output, synthetic_output, gp, lambda_gp):
        wasserstein_loss = tf.reduce_mean(synthetic_output) - tf.reduce_mean(real_output)
        return wasserstein_loss + lambda_gp * gp  # Include gradient penalty


class generator:

    def __init__(self, resolution):
        self.resolution = resolution
        self.build()

    def build(self, lr = 1e-4):
        self.model = tf.keras.Sequential()

        # Start with a dense layer
        self.model.add(tf.keras.layers.Dense(16 * 16 * 1024, use_bias = False, input_shape = (100,)))
        self.model.add(tf.keras.layers.Reshape((16, 16, 1024)))  # Starting at (16, 16, 1024)

        self.model.add(tf.keras.layers.Conv2DTranspose(512, (5, 5), strides = (2, 2), padding = 'same'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Conv2DTranspose(256, (5, 5), strides = (2, 2), padding = 'same'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides = (2, 2), padding = 'same'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = 'same'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Conv2DTranspose(32, (5, 5), strides = (2, 2), padding = 'same'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides = (2, 2), padding = 'same', use_bias = False, activation = 'tanh'))

        self.model.build()
        self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.5, beta_2 = 0.9)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = False)

    def loss(self, synthetic_output):
        return -tf.reduce_mean(synthetic_output)  # Wants to fool discriminator



if __name__ == "__main__":
    # Configure tensorflow
    tf.config.optimizer.set_jit(True)  # Use XLA computation

    # Initialize the parser
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

    snowgan.generate()
