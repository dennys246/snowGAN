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
        dataset_name = getattr(config, "dataset", None) or "rmdig/rocky_mountain_snowpack"
        self.dataset = load_dataset(dataset_name)
        self.config = config

        self.translator = {
            'core': 0,
            'profile': 1,
            'magnified_profile': 2,
            'crystal_card': 3
        }

        self.seen_profiles = []
        self.seen_cores = []

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

        datatype = self.translator[datatype]

        print(f"Collecting a batch...")
        
        while count < batch_size and self.config.train_ind < len(self.dataset['train']):
            sample = self.dataset['train'][self.config.train_ind]

            print(f"Checking sample at index {self.config.train_ind} - {sample['datatype']} - {datatype}")

            if sample['datatype'] == datatype:

                image = sample['image']

                scaled_image = self.preprocess_image(image)     

                batch.append(scaled_image)
                print(f"Image {self.config.train_ind} added")

                count += 1
            self.config.train_ind += 1

        if len(batch) > 0:
            return np.stack(batch)
        
        else:
            print("All images have been assessed and no available images could be found")
            return None
        
    def batch_merged(self, batch_size, config = None):
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

        print(f"Collecting a batch...")
        
        while count < batch_size and self.config.train_ind < len(self.dataset['train']):
            sample = self.dataset['train'][self.config.train_ind]

            print(f"Checking sample at index {self.config.train_ind} - {sample['datatype']}")

            # If we've found a core sample
            if sample['datatype'] == 0: 

                core_image = self.preprocess_image(sample['image'])
                self.seen_cores.append(self.config.train_ind)

                profile_ind = self.config.train_ind
                profile_image = None

                # Look for a magnified profile to pair with
                seen_profile = False
                while profile_image == None and profile_ind < len(self.dataset['train']):
                    profile_ind += 1

                    # If we've found a profile to use
                    if self.dataset['train'][profile_ind]['datatype'] == 1:
                        
                        seen_profile = True

                        # Check that it matches our core
                        if profile_ind in self.seen_profiles:
                            continue

                        if self.dataset['train'][profile_ind]['site'] != self.dataset['train'][self.config.train_ind]['site']:
                            continue

                        if self.dataset['train'][profile_ind]['column'] != self.dataset['train'][self.config.train_ind]['column']:
                            continue

                        if self.dataset['train'][profile_ind]['core'] != self.dataset['train'][self.config.train_ind]['core']:
                            continue

                        # Preprocessing image
                        profile_image = self.preprocess_image(self.dataset['train'][profile_ind]['image'])
                        self.seen_profiles.append(profile_ind)

                        break

                    # If all profiles have been viewed
                    if seen_profile and self.dataset['train'][profile_ind]['datatype'] == 0:
                        
                        break


                if profile_image == None:
                    print("Profile image not found, skipping...")
                    continue

                merged_image = self.merge_images(core_image, profile_image)

                batch.append(merged_image)
                print(f"Image add with core {self.config.train_ind} and profile {profile_ind} from segment {self.dataset['train'][profile_ind]['segment']}")

                count += 1
            self.config.train_ind += 1

        if len(batch) > 0:
            return np.stack(batch)
        
        else:
            print("All images have been assessed and no available images could be found")
            return None
        
    def preprocess_image(self, image):
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(np.array(image))  # Convert from PIL to tensor
        
        image = tf.image.resize(image, self.config.resolution)
        print(f"Max - {tf.reduce_max(image).numpy()} | Min {tf.reduce_min(image).numpy()} ")
        
        if image.shape.rank == 2:  # grayscale image
            image = tf.expand_dims(image, -1)

        scaled_image = (tf.cast(image, tf.float32) / 127.5) - 1.0  
        return scaled_image

    def merge_images(self, image_1, image_2):

        image_1_resized = tf.image.resize(image_1, (image_2.shape[0], image_2.shape[1]))

        merged_image = tf.concat([image_2, image_1_resized], axis=-1)

        return merged_image