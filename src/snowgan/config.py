import json, atexit, copy, os
from glob import glob

config_template = {
            "save_dir": None,
            "checkpoint": None,
            "dataset": "dennys246/rocky_mountain_snowpack",
            "datatype": "magnified_profile",
            "architecture": "generator",
            "resolution": (1024, 1024),
            "images": None,
            "trained_pool": None,
            "validation_pool": None,
            "test_pool": None,
            "model_history": None,
            "n_samples": 10,
            "epochs": 10,
            "current_epoch": 0,
            "batch_size": 8,
            "training_steps": 5,
            "learning_rate": 1e-4,
            "beta_1": 0.5,
            "beta_2": 0.9,
            "negative_slope": 0.25,
            "lambda_gp": 10.0,
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
    def __init__(self, config_filepath, config = None):
        if config is None:
            config = config_template

        self.config_filepath = config_filepath
        if os.path.exists(config_filepath): # Try and load config if folder passed in
            print(f"Loading config file: {self.config_filepath}")
            config_json = self.load_config(self.config_filepath)
            config_json['rebuild'] = False # Set to false if able to load a pre-existing model
        else:
            print("WARNING: Config not found, building from template...")
            config_json = copy.deepcopy(config)

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

    def configure(self, save_dir, checkpoint, dataset, datatype, architecture, resolution, images, trained_pool, validation_pool, test_pool, model_history, n_samples, epochs, current_epoch, batch_size, training_steps, learning_rate, beta_1, beta_2, negative_slope, lambda_gp, latent_dim, convolution_depth, filter_counts, kernel_size, kernel_stride, final_activation, zero_padding, padding, optimizer, loss, train_ind, trained_data, rebuild):
		# Process lists
        if isinstance(filter_counts, str):
            filter_counts = [int(datum) for datum in filter_counts.split(' ')]

        if isinstance(kernel_size, str):
            kernel_size = [int(datum) for datum in kernel_size.split(' ')]

        if isinstance(kernel_stride, str):
            kernel_stride = [int(datum) for datum in kernel_stride.split(' ')]
        
        #-------------------------------- Model Set-Up -------------------------------#
        self.save_dir = save_dir or "keras/snowgan/"
        self.checkpoint = checkpoint or "generator.keras"
        self.dataset = dataset or "dennys246/rocky_mountain_snowpack"
        self.datatype = datatype or "magnified_profile"
        self.architecture = architecture or "generator"
        self.resolution = resolution or (1024, 1024)
        self.images = images or None 
        self.trained_pool = trained_pool or None
        self.validation_pool = validation_pool or None
        self.test_pool = test_pool or None
        self.model_history = model_history or None
        self.n_samples = n_samples or 10
        self.epochs = epochs or 10
        self.current_epoch = current_epoch or 0
        self.batch_size = batch_size or 8
        self.training_steps = training_steps or 5
        self.learning_rate = learning_rate or 1e-4
        self.beta_1 = beta_1 or 0.5
        self.beta_2 = beta_2 or 0.9
        self.negative_slope = negative_slope or 0.25
        self.lambda_gp = lambda_gp or None
        self.latent_dim = latent_dim or 100
        self.convolution_depth = convolution_depth or 5
        self.filter_counts = filter_counts or [64, 128, 256, 512, 1024]
        self.kernel_size = kernel_size or [5, 5]
        self.kernel_stride = kernel_stride or [2, 2]
        self.final_activation = final_activation or "tanh"
        self.zero_padding = zero_padding or None
        self.padding = padding or "same"
        self.optimizer = optimizer or "adam"
        self.loss = loss or None
        self.train_ind = train_ind or 0
        self.trained_data = trained_data or []
        self.rebuild = rebuild or False

    def dump(self):
        config = {
            "save_dir": self.save_dir,
            "checkpoint": self.checkpoint,
            "dataset": self.dataset,
            "datatype": self.datatype,
            "architecture": self.architecture,
            "resolution": self.resolution,
            "images": self.images,
            "trained_pool": self.trained_pool,
            "validation_pool": self.validation_pool,
            "test_pool": self.test_pool,
            "model_history": self.model_history,
            "n_samples": self.n_samples,
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

def load_gen_config(config_filepath, config = None):
    # Configure the discriminator
    gen_config = build(config_filepath, config)

    if not os.path.exists(config_filepath):
        split = config_filepath.split("/")
        gen_config.save_dir = gen_config.save_dir or "/".join(split[:-1]) + "/"
        gen_config.checkpoint = gen_config.checkpoint or "generator.keras"
        gen_config.architecture = "generator"
    return gen_config 

def configure_gen(config, args):
    config = configure_generic(config, args)

    config['kernel_size'] = args.gen_kernel
    config['kernel_stride'] = args.gen_stride
    config['learning_rate'] = args.gen_lr
    config['beta_1'] = args.gen_beta_1
    config['beta_2'] = args.gen_beta_2
    config['negative_slope'] = args.gen_negative_slope
    config['training_steps'] = args.gen_steps
    config['filter_counts'] = args.gen_filters
    return config
    

def load_disc_config(config_filepath, config = None):
    # Configure the discriminator
    disc_config = build(config_filepath, config)

    if not os.path.exists(config_filepath):
        split = config_filepath.split("/")
        disc_config.save_dir = disc_config.save_dir or "/".join(split[:-1]) + "/"
        disc_config.checkpoint = disc_config.checkpoint or "discriminator.keras"
        disc_config.architecture = "discriminator"
    return disc_config 

def configure_disc(config, args):
    config = configure_generic(config, args)

    config['kernel_size'] = args.disc_kernel
    config['kernel_stride'] = args.disc_stride
    config['learning_rate'] = args.disc_lr
    config['beta_1'] = args.disc_beta_1
    config['beta_2'] = args.disc_beta_2
    config['negative_slope'] = args.disc_negative_slope
    config['training_steps'] = args.disc_steps
    config['filter_counts'] = args.disc_filters
    return config

def configure_generic(config, args):
    config['save_dir'] = args.save_dir
    config['checkpoint'] = args.checkpoint
    config['rebuild'] = args.rebuild

    config['resolution'] = args.resolution
    config['n_samples'] = args.n_samples
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['latent_dim'] = args.latent_dim
    return config

