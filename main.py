import argparse, os, datasets
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.trainer import Trainer
from src.generate import generate, make_movie
import src.config

def configure_device(args):
    # Configure tensorflow
    if hasattr(args, "device") and args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # Enable memory growth for all GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.xla == True: # Use XLA computation for faster runtime operations
        tf.config.optimizer.set_jit(True)  
    if args.mixed_precision == True : # Use mixed precision for faster training
        set_global_policy("mixed_float16")

def configure_model(args):
    gen_config = src.config.build(f"{args.save_dir}generator_config.json")
    gen_config.save_dir = args.save_dir
    gen_config.architecture = "generator"
    gen_config.resolution = args.resolution
    gen_config.epochs = args.epochs 
    gen_config.batch_size = args.batch_size
    gen_config.kernel_size = args.gen_kernel
    gen_config.kernel_stride = args.gen_stride
    gen_config.learning_rate = args.gen_lr
    gen_config.beta_1 = args.gen_beta_1
    gen_config.beta_2 = args.gen_beta_2
    gen_config.negative_slope = args.gen_negative_slope
    gen_config.training_steps = args.gen_steps
    gen_config.filter_counts = args.gen_filters
    gen_config.latent_dim = args.latent_dim

    disc_config = src.config.build(f"{args.save_dir}discriminator_config.json")
    disc_config.save_dir = args.save_dir
    disc_config.architecture = "discriminator"
    disc_config.resolution = args.resolution
    disc_config.epochs = args.epochs 
    disc_config.batch_size = args.batch_size
    disc_config.kernel_size = args.disc_kernel
    disc_config.kernel_stride = args.disc_stride
    disc_config.learning_rate = args.disc_lr
    disc_config.beta_1 = args.disc_beta_1
    disc_config.beta_2 = args.disc_beta_2
    disc_config.negative_slope = args.disc_negative_slope
    disc_config.lambda_gp = args.disc_lambda_gp
    disc_config.training_steps = args.disc_steps
    disc_config.filter_counts = args.disc_filters
    disc_config.latent_dim = args.latent_dim

    return gen_config, disc_config 

def load_dataset(dataset_dir):
    return datasets.load_dataset(dataset_dir)

def main():

    # Initialize the parser for accepting arugments into a command line call
    parser = argparse.ArgumentParser(description = "The snowGAN model is used to train a GAN on a dataset of snow samples magnified on a crystal card. You can define how the model runs by the number of epochs, batch sizes and other parameters. You can also pass in a path to a pre-trained snowGAN to accomplish transfer learning on new GAN tasks!")

    # Add command-line arguments
    parser.add_argument('--mode', type = str, choices = ["train", "generate"], required = True, help = "Mode to run the model in, either generate fake data or train the model")
    parser.add_argument('--dataset_dir', type = str, default = 'rmdig/rocky_mountain_snowpack', help = "Path to the Rocky Mountain Snowpack dataset, if none provided it will download directly from HF remote repository")
    parser.add_argument('--save_dir', type = str, default = "keras/snowgan/", help = "Path to save results where a pre-trained model may be found (defaults to keras/snowgan/)")
    parser.add_argument('--new', type = bool, default = False, help = 'Whether to rebuild model from scratch (defaults to False)')
    
    parser.add_argument('--device', type = str, choices = ["cpu", "gpu"], default = "gpu", help = 'Device to run the model on (defaults to gpu)')
    parser.add_argument('--xla', type = bool, default = False, help = 'Whether to use accelerated linear algebra (XLA) (defaults to False)')
    parser.add_argument('--mixed_precision', type = bool, default = False, help = 'Whether to use mixed precision training (defaults to False)')

    parser.add_argument('--resolution', type = set, default = (1024, 1024), help = 'Resolution to downsample images too (Default set to (1024, 1024))')
    parser.add_argument('--synthetics', type = int, default = 10, help = "Number of synthetic images to generate (defaults to 10)")
    parser.add_argument('--batch_size', type = int, default = 8, help = 'Batch size (Defaults to 8)')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Epochs to train on (Defaults to 10)')
    parser.add_argument('--latent_dim', type = float, default = 100, help = 'Latent dimension size (Defaults to 100)')

    parser.add_argument('--gen_kernel', type = list, default = [10, 10], help = 'Generator kernel size (Defaults to [5, 5])')
    parser.add_argument('--gen_stride', type = list, default = [3, 3], help = 'Generator kernel stride (Defaults to [2, 2])')
    parser.add_argument('--gen_lr', type = float, default = 1e-3, help = 'Generators optimizer learning rate (Defaults to 0.001)')
    parser.add_argument('--gen_beta_1', type = float, default = 0.5, help = 'Generators optimizer adam beta one (Defaults to 0.5)')
    parser.add_argument('--gen_beta_2', type = float, default = 0.9, help = 'Generators optimizer adam beta two (Defaults to 0.9)')
    parser.add_argument('--gen_negative_slope', type = float, default = 0.25, help = 'Generators negative slope for leaky relu (Defaults to 0.25)')
    parser.add_argument('--gen_steps', type = int, default = 3, help = 'Training steps the generator takes per batch (Defaults to 5)')
    parser.add_argument('--gen_filters', type = list, default = [1024, 512, 256, 128, 64], help = 'Generators filters per convolution layer (Defaults to [1024, 512, 256, 128, 64])')
    
    parser.add_argument('--disc_kernel', type = list, default = [10, 10], help = 'Discriminator kernel size (Defaults to [5, 5])')
    parser.add_argument('--disc_stride', type = list, default = [3, 3], help = 'Discriminator kernel stride (Defaults to [2, 2])')
    parser.add_argument('--disc_lr', type = float, default = 1e-4, help = 'Discriminators learning rate (Defaults to 0.0001)')
    parser.add_argument('--disc_beta_1', type = float, default = 0.5, help = 'Discriminators adam beta one (Defaults to 0.5)')
    parser.add_argument('--disc_beta_2', type = float, default = 0.9, help = 'Discriminators dam beta two (Defaults to 0.9)')
    parser.add_argument('--disc_negative_slope', type = float, default = 0.25, help = 'Discriminators negative slope for leaky relu (Defaults to 0.25)')
    parser.add_argument('--disc_lambda_gp', type = int, default = 10.0, help = 'Disciminators lambda GP (Defaults to 10.0)')
    parser.add_argument('--disc_steps', type = int, default = 1, help = 'Steps the discriminator takes per batch (Defaults to 1)')
    parser.add_argument('--disc_filters', type = list, default = [64, 128, 256, 512, 1024], help = 'Discriminator filters per convolution layer (Defaults to [64, 128, 256, 512, 1024])')

    # Parse the arguments
    args = parser.parse_args()

    configure_device(args)

    gen_config, disc_config = configure_model(args)

    # Load the generator
    generator = Generator(gen_config)
    generator.model.build((args.latent_dim,))

    if args.mode == "train":
        dataset = load_dataset(args.dataset_dir)

        # Load the discriminator
        discriminator = Discriminator(disc_config)
        discriminator.model.build((args.resolution[0], args.resolution[1], 3))

        # Call to trainer
        trainer = Trainer(generator, discriminator)
        trainer.train(batch_size = args.batch_size, epochs = args.epochs)
    
    if args.mode == "generate":

        generate(generator, args.synthetics, args.latent_dim, args.save_dir)

if __name__ == "__main__":

    main()

