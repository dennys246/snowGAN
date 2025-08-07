import json, atexit, copy, os
from glob import glob

config_template = {
            "save_dir": "outputs",
            "dataset": "dennys246/rocky_mountain_snowpack",
            "architecture": "generator",
            "resolution": (1024, 1024),
            "images": None,
            "trained_pool": None,
            "validation_pool": None,
            "test_pool": None,
            "model_history": None,
            "synthetics": 10,
            "epochs": 10,
            "current_epoch": 0,
            "batch_size": 8,
            "training_steps": 5,
            "learning_rate": 1e-4,
            "beta_1": 0.5,
            "beta_2": 0.9,
            "negative_slope": 0.25,
            "lambda_gp": None,
            "latent_dim": 100,
            "convolution_depth": 5,
            "filter_counts": [64, 128, 256, 512, 1024],
            "kernel_size": [5, 5],
            "kernel_stride": (2, 2),
            "final_activation": "tanh",
            "zero_padding": None,
            "padding": "same",
            "optimizer": "adam",
            "loss": None,
            "train_ind": 0,
            "trained_data": [],
            "rebuild": False
}

class build:
    def __init__(self, config_filepath):
        self.config_filepath = config_filepath
        if os.path.exists(config_filepath): # Try and load config if folder passed in
            print(f"Loading config file: {self.config_filepath}")
            config_json = self.load_config(self.config_filepath)
            config_json['rebuild'] = False # Set to false if able to load a pre-existing model
        else:
            print("WARNING: Config not found, building from template...")
            config_json = copy.deepcopy(config_template)

        self.configure(**config_json) # Build configuration

        atexit.register(self.save_config)

        
    def __repr__(self):
        return '\n'.join([f"{key}: {value}" for key, value in self.__dict__.items()])
    
    def save_config(self, config_filepath = None):
        # Save the config filepath if passed in
        if config_filepath: self.config_filepath = config_filepath

        if os.path.exists(os.path.basename(self.config_filepath)) == False : # Make directory if necessary
            os.makedirs(os.path.dirname(self.config_filepath), exist_ok=True)

        with open(self.config_filepath, 'w') as config_file:
            json.dump(self.dump(), config_file, indent = 4)
             
    def load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config_json = json.load(config_file)
        else:
            config_json = config_template
        return config_json

    def configure(self, save_dir, dataset, architecture, resolution, images, trained_pool, validation_pool, test_pool, model_history, synthetics, epochs, current_epoch, batch_size, training_steps, learning_rate, beta_1, beta_2, negative_slope, lambda_gp, latent_dim, convolution_depth, filter_counts, kernel_size, kernel_stride, final_activation, zero_padding, padding, optimizer, loss, train_ind, trained_data, rebuild):
		#-------------------------------- Model Set-Up -------------------------------#
        self.save_dir = save_dir
        self.dataset = dataset
        self.architecture = architecture
        self.resolution = resolution
        self.images = images
        self.trained_pool = trained_pool
        self.validation_pool = validation_pool
        self.test_pool = test_pool
        self.model_history = model_history
        self.synthetics = synthetics
        self.epochs = epochs
        self.current_epoch = current_epoch
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.negative_slope = negative_slope
        self.lambda_gp = lambda_gp
        self.latent_dim = latent_dim
        self.convolution_depth = convolution_depth
        self.filter_counts = filter_counts
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.final_activation = final_activation
        self.zero_padding = zero_padding
        self.padding = padding
        self.optimizer = optimizer
        self.loss = loss
        self.train_ind = train_ind
        self.trained_data = trained_data
        self.rebuild = rebuild

    def dump(self):
        config = {
            "save_dir": self.save_dir,
            "dataset": self.dataset,
            "architecture": self.architecture,
            "resolution": self.resolution,
            "images": self.images,
            "trained_pool": self.trained_pool,
            "validation_pool": self.validation_pool,
            "test_pool": self.test_pool,
            "model_history": self.model_history,
            "synthetics": self.synthetics,
            "epochs": self.epochs,
            "current_epoch": self.current_epoch,
            "batch_size": self.batch_size,
            "training_steps": self.training_steps,
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "negative_slope": self.negative_slope,
            "lambda_gp": self.lambda_gp,
            "latent_dim": self.latent_dim,
            "convolution_depth": self.convolution_depth,
            "filter_counts": self.filter_counts,
            "kernel_size": self.kernel_size,
            "kernel_stride": self.kernel_stride,
            "final_activation":self.final_activation,
            "zero_padding": self.zero_padding,
            "padding": self.padding,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "train_ind": self.train_ind,
            "trained_data": self.trained_data,
            "rebuild": self.rebuild
        }
        return config
