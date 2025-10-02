import os, argparse, datasets
from tensorflow.keras.mixed_precision import set_global_policy
import tensorflow as tf
from pathlib import Path

from snowgan.config import build as configuration

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


def parse_args():
        # Initialize the parser for accepting arugments into a command line call
    parser = argparse.ArgumentParser(description = "The snowGAN model is used to train a GAN on a dataset of snow samples magnified on a crystal card. You can define how the model runs by the number of epochs, batch sizes and other parameters. You can also pass in a path to a pre-trained snowGAN to accomplish transfer learning on rebuild GAN tasks!")

    # Add command-line arguments
    parser.add_argument('--mode', type = str, choices = ["train", "generate"], required = True, help = "Mode to run the model in, either generate fake data or train the model")
    parser.add_argument('--dataset_dir', type = str, default = 'rmdig/rocky_mountain_snowpack', help = "Path to the Rocky Mountain Snowpack dataset, if none provided it will download directly from HF remote repository")
    parser.add_argument('--save_dir', type = str, default = "keras/snowgan/", help = "Path to save results where a pre-trained model may be found (defaults to keras/snowgan/)")
    
    parser.add_argument('--rebuild', type = bool, default = False, help = 'Whether to rebuild model from scratch (defaults to False)')
    
    parser.add_argument('--device', type = str, choices = ["cpu", "gpu"], default = "gpu", help = 'Device to run the model on (defaults to gpu)')
    parser.add_argument('--xla', type = bool, default = False, help = 'Whether to use accelerated linear algebra (XLA) (defaults to False)')
    parser.add_argument('--mixed_precision', type = bool, default = False, help = 'Whether to use mixed precision training (defaults to False)')

    parser.add_argument('--resolution', type = set, help = 'Resolution to downsample images too (Default set to (1024, 1024))')
    parser.add_argument('--n_samples', type = int, default = 10, help = "Number of synthetic images to generate (defaults to 10)")
    parser.add_argument('--batch_size', type = int, default = 8, help = 'Batch size (Defaults to 8)')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Epochs to train on (Defaults to 10)')
    parser.add_argument('--latent_dim', type = float, default = 100, help = 'Latent dimension size (Defaults to 100)')

    parser.add_argument('--gen_checkpoint', type = str, help = "Path to a pre-trained generator model to load")
    parser.add_argument('--gen_kernel', type = list, help = 'Generator kernel size (Defaults to [5, 5])')
    parser.add_argument('--gen_stride', type = list, help = 'Generator kernel stride (Defaults to [2, 2])')
    parser.add_argument('--gen_lr', type = float, help = 'Generators optimizer learning rate (Defaults to 0.001)')
    parser.add_argument('--gen_beta_1', type = float, help = 'Generators optimizer adam beta one (Defaults to 0.5)')
    parser.add_argument('--gen_beta_2', type = float, help = 'Generators optimizer adam beta two (Defaults to 0.9)')
    parser.add_argument('--gen_negative_slope', type = float, help = 'Generators negative slope for leaky relu (Defaults to 0.25)')
    parser.add_argument('--gen_steps', type = int, help = 'Training steps the generator takes per batch (Defaults to 5)')
    parser.add_argument('--gen_filters', type = list, help = 'Generators filters per convolution layer (Defaults to [1024, 512, 256, 128, 64])')
    
    parser.add_argument('--disc_checkpoint', type = str, help = "Path to a pre-trained discriminator model to load")
    parser.add_argument('--disc_kernel', type = list, help = 'Discriminator kernel size (Defaults to [5, 5])')
    parser.add_argument('--disc_stride', type = list, help = 'Discriminator kernel stride (Defaults to [2, 2])')
    parser.add_argument('--disc_lr', type = float, help = 'Discriminators learning rate (Defaults to 0.0001)')
    parser.add_argument('--disc_beta_1', type = float, help = 'Discriminators adam beta one (Defaults to 0.5)')
    parser.add_argument('--disc_beta_2', type = float, help = 'Discriminators dam beta two (Defaults to 0.9)')
    parser.add_argument('--disc_negative_slope', type = float, help = 'Discriminators negative slope for leaky relu (Defaults to 0.25)')
    parser.add_argument('--disc_lambda_gp', type = int, help = 'Disciminators lambda GP (Defaults to 10.0)')
    parser.add_argument('--disc_steps', type = int, help = 'Steps the discriminator takes per batch (Defaults to 1)')
    parser.add_argument('--disc_filters', type = list, help = 'Discriminator filters per convolution layer (Defaults to [64, 128, 256, 512, 1024])')

    # Parse the arguments
    return parser.parse_args()

def get_repo_root(start: str = ".") -> Path:
    path = Path(start).resolve()
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("No .git directory found")
