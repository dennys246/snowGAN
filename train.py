import os, generate, snowgan
import tensorflow as tf
from matplotlib import pyplot as plt
from glob import glob

class trainer:

    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

        self.loss = {'disc': [], 'gen': []}

        self.pipe = pipeline(self.path, self.resolution)

        self.synthetics = 10 # Number of synthetic images to generate after training

        self.batch_size = 8 # Number of images to load in per training batch
        self.epochs = 50 # Number of training epochs per training batch

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
        if not batches: batches = int(round(len(self.pipe.avail_photos)/batch_size, 0))
        if batch_size: self.batch_size = batch_size
        if epochs: self.epochs = epochs

        # Assess current training progress of the model
        previous_batches = len(glob(f"{self.path}/synthetic_images/batch_*/"))

        # Increment batches to account for past training history
        batches += previous_batches

        # Iterate through requested training batches 
        for batch in range(previous_batches, batches):

            self.x = self.pipe.load_batch(self.batch_size) # Load a new batch of subjects

            print(f"Training on batch {batch + 1}...")
            for epoch in range(self.epochs): # Iterate through each requested epoch
                self.learn(self.x) # Train on batch of images
                print(f'Epoch {epoch + 1} | Generator loss: {round(float(self.loss['gen'][-1]), 3)} | Discrimintator loss: {round(float(self.loss['disc'][-1]), 3)} |')
                
                self.plot_history() # Update history with progress
            
            # Generate synthetic images to the batch folder to track progress
            _ = self.generate(f"synthetic_batch_{batch}", f"Synthetic Image after {round((float(batch)/float(batches))*100, 2)}%", f"batch_{batch}/")
            
            # Save the models state
            self.save_model(f"{self.path}/synthetic_images/batch_{batch}") # Need to consider more dynamic way to do this and remove old history

    def save_model(self, path = None):
        """
        Save the currently loaded generator and discriminator to a given path

        Function arguments:
            path (str) - Folder to save the generator and discriminator .keras files
        """
        # Update path if none passed in
        if path == None:
            path = self.path
        
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
            path = self.path

        # Load generator and discriminator .keras files
        self.model.gen.model = tf.keras.models.load_model(f"{path}/generator.keras")
        self.model.disc.model = tf.keras.models.load_model(f"{path}/discriminator.keras")
        print(f"Models loaded from {path}...")

    def log_history(self):
        """
        Plot and save the generator and discriminator history loaded in the snowGAN object
        """

        # Save the current generate loss progress
        with open(f"{self.path}generator_loss.txt", "w") as file:
            for loss in self.loss['gen']:
                file.write(f"{loss}\n")

        # Save the current discriminator loss
        with open(f"{self.path}discriminator_loss.txt", "w") as file:
            for loss in self.loss['disc']:
                file.write(f"{loss}\n")

    def plot_history(self):
        # Plot the generator and discriminator loss history
        plt.plot(self.loss['gen'], label = 'Generator loss')
        plt.plot(self.loss['disc'], label = 'Discriminator loss')
        plt.title("GAN History")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.savefig(f'{self.path}history.png')
        plt.close()