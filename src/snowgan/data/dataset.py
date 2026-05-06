from functools import cached_property

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
        manifest_df = self.dataset["train"].to_pandas().drop(columns=["image", "audio"], errors="ignore")
        self.manifest_columns = manifest_df.columns.tolist()
        self.manifest = manifest_df.values.tolist()

        self.config = config

        self.translator = {
            'core': 0,
            'profile': 1,
            'magnified_profile': 2,
            'crystal_card': 3
        }

        # Track seen profiles across runs/epochs and keep in sync with config for persistence
        self.seen_profiles = set(getattr(self.config, "seen_profiles", []) or [])
        self.config.seen_profiles = self.seen_profiles
        self.seen_cores = set()

    @cached_property
    def pair_index(self):
        """Cross-pair index of the manifest, grouped by (site, column, core).

        Returns:
            dict[tuple, list[tuple[int, int]]] — keys are ``(site, column, core)``
            tuples for every group that has at least one core (datatype=0) and at
            least one magnified profile (datatype=2). Values are the full Cartesian
            product ``[(core_idx, profile_idx), ...]`` of every core row index ×
            every magnified-profile row index that share the group key.

        Intended for downstream consumers (e.g. AvAI transfer learning) that split
        at the group level to prevent leakage and want every cross-pair per group
        as a combinatorial augmentation. The GAN trainer's ``batch_merged`` does
        not use this — it has different (per-epoch, no-reuse) pairing semantics.

        Read-only and side-effect-free: does not touch ``self.config``,
        ``self.seen_profiles``, ``self.seen_cores``, or ``self.config.train_ind``.
        Cached on the instance after the first access.
        """
        cores: dict[tuple, list[int]] = {}
        profiles: dict[tuple, list[int]] = {}
        for idx in range(len(self.manifest)):
            meta = self._get_manifest_entry(idx)
            if meta is None:
                continue
            datatype = meta.get("datatype")
            if datatype != 0 and datatype != 2:
                continue
            key = (meta.get("site"), meta.get("column"), meta.get("core"))
            bucket = cores if datatype == 0 else profiles
            bucket.setdefault(key, []).append(idx)

        return {
            key: [(c, p) for c in cores[key] for p in profiles[key]]
            for key in cores.keys() & profiles.keys()
        }

    def _get_manifest_entry(self, index):
        if index < 0 or index >= len(self.manifest):
            return None
        row = self.manifest[index]
        return {key: row[i] for i, key in enumerate(self.manifest_columns)}

    def reset_seen_profiles(self):
        self.seen_profiles.clear()
        self.config.seen_profiles = self.seen_profiles

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
        
        while count < batch_size and self.config.train_ind < len(self.manifest):
            meta = self._get_manifest_entry(self.config.train_ind)
            if meta is None:
                break
            sample_datatype = meta.get("datatype")

            print(f"Checking sample at index {self.config.train_ind} - {sample_datatype} - {datatype}")

            if sample_datatype == datatype:
                sample = self.dataset['train'][self.config.train_ind]

                image = sample['image']

                scaled_image = self.preprocess_image(image)

                # Add a depth dimension for single-view samples
                scaled_image = tf.expand_dims(scaled_image, axis=0)
                if hasattr(self.config, "depth"):
                    self.config.depth = int(scaled_image.shape[0])

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
        
        while count < batch_size and self.config.train_ind < len(self.manifest):
            meta = self._get_manifest_entry(self.config.train_ind)
            if meta is None:
                break
            sample_datatype = meta.get("datatype")

            print(f"Checking sample at index {self.config.train_ind} - {sample_datatype}")

            # If we've found a core sample
            if sample_datatype == 0: 

                sample = self.dataset['train'][self.config.train_ind]

                core_image = self.preprocess_image(sample['image'])
                self.seen_cores.add(self.config.train_ind)

                profile_ind = self.config.train_ind
                profile_image = None

                # Look for a magnified profile to pair with
                seen_profile = False
                while profile_image == None and (profile_ind + 1) < len(self.manifest):
                    profile_ind += 1
                    profile_meta = self._get_manifest_entry(profile_ind)

                    if profile_ind >= len(self.manifest):
                        break
                    if profile_meta is None:
                        break

                    # If we've found a profile to use
                    if profile_meta.get("datatype") == 2:
                        
                        seen_profile = True

                        # Check that it matches our core
                        if profile_ind in self.seen_profiles:
                            continue

                        if profile_meta.get('site') != sample['site']:
                            continue

                        if profile_meta.get('column') != sample['column']:
                            continue

                        if profile_meta.get('core') != sample['core']:
                            continue

                        # Preprocessing image
                        profile_sample = self.dataset['train'][profile_ind]
                        profile_image = self.preprocess_image(profile_sample['image'])
                        self.seen_profiles.add(profile_ind)
                        self.config.seen_profiles = self.seen_profiles

                    # If all profiles have been viewed
                    if seen_profile and profile_meta.get("datatype") == 0:
                        
                        break


                if profile_image == None:
                    self.config.train_ind += 1
                    
                    print("Profile image not found, skipping...")
                    continue

                merged_image = self.merge_images(core_image, profile_image)

                # Track stacked depth (views)
                if hasattr(self.config, "depth"):
                    depth = merged_image.shape[0]
                    if depth is not None:
                        self.config.depth = int(depth)

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

        # Resize first image to match spatial dims then stack along a new depth axis
        image_1_resized = tf.image.resize(image_1, (image_2.shape[0], image_2.shape[1]))
        merged_image = tf.stack([image_2, image_1_resized], axis=0)  # depth x H x W x C
        return merged_image
