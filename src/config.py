import json, atexit, copy
from glob import glob

config_template = {
            "save_dir": "outputs"
            "dataset": "dennys246/rocky_mountain_snowpack",
            "architecture": "generator",
            "resolution": (1024, 1024),
            "images": None,
            "trained_pool": None,
            "validation_pool": None,
            "test_pool": None,
            "model_history": None,
            "epochs": 10,
            "current_epoch": 0,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "beta_1": 0.5,
            "beta_2": 0.9,
            "alpha": 0.25,
            "latent_dim": 100,
            "training_steps": 5
            "convolution_depth": 5,
            "filter_counts": [64, 128, 256, 512, 1024],
            "kernel_size": [5, 5],
            "kernel_stride": (2, 2),
            "final_activation": "tahn"
            "padding": "same",
            "optimizer": "adam",
            "loss": None,
            "rebuild": False
}

class build:
    def __init__(self, config_filepath):
        if os.path.exists(config_filepath): # Try and load config if folder passed in
            print(f"Loading config file: {config_filepath}")
            config_json = self.load_config(config_filepath)
            config_json['rebuild'] = False # Set to false if able to load a pre-existing model
        else:
            print("WARNING: Config not found, building from template...")
            config_json = copy.deepcopy(config_template)

        self.configure(**config_json) # Build configuration

        atexit.register(self.save_config)
        
    def __repr__(self):
        return '\n'.join([f"{key}: {value}" for key, value in self.__dict__.items()])
    
    def save_config(self, config_path):
        with open(config_path, 'w') as config_file:
            json.dump(self.dump(), config_file, indent = 4)
             
    def load_config(self, config_path):
        with open(config_path, "r") as config_file:
            config_json = json.load(config_file)
        return config_json

    def configure(self, save_dir, dataset, architecture, resolution, images, trained_pool, validation_pool, test_pool, model_history, epochs, current_epoch, batch_size, learning_rate, beta_1, beta_2, alpha, latent_dim, convolution_depth, init_filter_count, kernel_size, kernel_stride, zero_padding, padding, pool_size, pool_stride, optimizer, loss, rebuild):
		#-------------------------------- Model Set-Up -------------------------------#
        self.save_dir = save_dir
        self.dataset = dataset
        self.architecture = architecture
        self.resolution
        self.images = images
        self.trained_pool = trained_pool
        self.validation_pool = validation_pool
        self.test_pool = test_pool
        self.model_history = model_history
        self.epochs = epochs
        self.current_epoch = current_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha,	
        self.latent_dim,
        self.convolution_depth = convolution_depth
        self.filter_counts = init_filter_count
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.zero_padding = zero_padding
        self.padding = padding
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.optimizer = optimizer
        self.loss = loss
        self.rebuild = rebuild

    def dump(self):
        config = {
            "save_dir": self.save_dir
            "dataset": self.dataset,
            "architecture": self.architecture,
            "resolution": self.resolution,
            "images": self.images,
            "trained_pool": self.trained_pool,
            "validation_pool": self.validation_pool,
            "test_pool": self.test_pool,
            "model_history": self.model_history,
            "epochs": self.epochs,
            "current_epoch": self.current_epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "alpha": self.alpha,
            "latent_dim", self.latent_dim
            "convolution_depth": self.convolution_depth,
            "filter_counts": self.filter_counts,
            "kernel_size": self.kernel_size,
            "kernel_stride": self.kernel_stride,
            "zero_padding": self.zero_padding,
            "padding": self.padding,
            "pool_size": self.pool_size,
            "pool_stride": self.pool_stride,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "rebuild": self.rebuild
        }
        return config
