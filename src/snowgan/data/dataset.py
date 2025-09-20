from datasets import load_dataset
import tensorflow as tf
import numpy as np

class DataManager:

    def __init__(self, config):
        """
        Initialize the DataManager with a configuration object.
        
        Function arguments:
            config (Config) - Configuration object containing dataset parameters
            
        This class handles loading the dataset and preparing batches of data for training.
        """
        self.dataset = load_dataset("rmdig/rocky_mountain_snowpack")
        self.config = config

    def batch(self, batch_size, datatype = "magnified_profile", config = None):
        """
        Prepare a batch of data from the dataset.
        
        Function arguments:
            batch_size (int) - Number of samples to include in the batch
            datatype (int) - Type of data to load (0: core, 1:
                        profile, 2: magnified_profile, 3: crystal_card)
            config (Config) - Optional configuration object to override current settings
        
        Returns:
            np.ndarray - Batch of images ready for training"""
        count = 0
        batch = []

        if config:
            self.config = config

        translator = {
            'core': 0,
            'profile': 1,
            'magnified_profile': 2,
            'crystal_card': 3
        }

        datatype = translator[datatype]

        print(f"Collecting a batch...")
        
        while count < batch_size and self.config.train_ind < len(self.dataset['train']):
            sample = self.dataset['train'][self.config.train_ind]

            print(f"Checking sample at index {self.config.train_ind} - {sample['datatype']} - {datatype}")

            if sample['datatype'] == datatype:

                image = sample['image']

                if not isinstance(image, tf.Tensor):
                    image = tf.convert_to_tensor(np.array(image))  # Convert from PIL to tensor
                
                image = tf.image.resize(image, self.config.resolution)
                print(f"Max - {tf.reduce_max(image).numpy()} | Min {tf.reduce_min(image).numpy()} ")
                
                if image.shape.rank == 2:  # grayscale image
                    image = tf.expand_dims(image, -1)

                scaled_image = (tf.cast(image, tf.float32) / 127.5) - 1.0         

                batch.append(scaled_image)
                print(f"Image {self.config.train_ind} added")

                count += 1
            self.config.train_ind += 1

        if len(batch) > 0:
            return np.stack(batch)
        
        else:
            print("All images have been assessed and no available images could be found")
            return None
