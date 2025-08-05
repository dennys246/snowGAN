
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.trainer import Trainer
from src.generate import generate, make_movie
import src.config


def configure_device(args):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if gpus: # If GPU's available
                for gpu in gpus: # Set memory growth to prevent TensorFlow from using all memory
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus[0].name}")
            else: # Use only CPU's
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
        except RuntimeError as e:
            print("Failed to set GPU memory growth:", e)
    else:
        print("No GPU found. Using CPU.")
    # Configure tensorflow
    if args.xla == True: # Use XLA computation for faster runtime operations
        tf.config.optimizer.set_jit(True)  

def configure_model(args):
    gen_config = src.config.build(f"{args.save_dir}generator_config.json")
    gen_config['save_dir'] = args.save_dir
    gen_config['architecture'] = "generator"
    gen_config['resolution'] = args.resolution
    gen_config['synthetics'] = args.synthetics
    gen_config['epochs'] = args.epochs 
    gen_config['batch_size'] args.batch_size
    gen_config['batches'] args.batches
    gen_config['learning_rate'] = args.gen_lr
    gen_config['beta_1'] = args.gen_beta_1
    gen_config['beta_2'] = args.gen_beta_2
    gen_config['training_steps'] = args.gen_steps
    gen_config['latent_dim'] = args.latent_dim

    disc_config = src.config.build(f"{args.save_dir}discriminator_config.json")
    disc_config['save_dir'] = args.save_dir
    disc_config['architecture'] = "discriminator"
    disc_config['resolution'] = args.resolution
    disc_config['synthetics'] = args.synthetics
    disc_config['epochs'] = args.epochs 
    disc_config['batch_size'] args.batch_size
    disc_config['batches'] args.batches
    disc_config['learning_rate'] = args.disc_lr
    disc_config['beta_1'] = args.disc_beta_1
    disc_config['beta_2'] = args.disc_beta_2
    disc_config['training_steps'] = args.disc_steps
    disc_config['latent_dim'] = args.latent_dim

    return gen_config, disc_config

def load_dataset(batch_size, image_size):
    dataset = datasets.load_dataset("rmdig/rocky_mountain_snowpack")
    return dataset

def main(args):

    # Initialize the parser for accepting arugments into a command line call
    parser = argparse.ArgumentParser(description = "The snowGAN model is used to train a GAN on a dataset of snow samples magnified on a crystal card. You can define how the model runs by the number of epochs, batch sizes and other parameters. You can also pass in a path to a pre-trained snowGAN to accomplish transfer learning on new GAN tasks!")

    # Add command-line arguments
    parser.add_argument('--mode', type = str, choices = ["train", "generate"], required = True, help = "Mode to run the model in, either generate fake data or train the model")
    parser.add_argument('--save_dir', type = str, default = "outputs/", help = "Path to a pre-trained model or directory to save results (defaults to outputs/)")
    parser.add_argument('--resolution', type = set, default = (1024, 1024), help = 'Resolution to downsample images too (Default set to (1024, 1024))')
    parser.add_argument('--synthetics', type = int, default = 10, help = "Number of synthetic images to generate (defaults to 10)")
    parser.add_argument('--batches', type = int, default = None, help = 'Number of batches to run (Default to max available)')
    parser.add_argument('--batch_size', type = int, default = 8, help = 'Batch size (Defaults to 8)')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Epochs to train on (Defaults to 10)')
    parser.add_argument('--gen_lr', type = float, default = 1e-4, help = 'Generators optimizer learning rate (Defaults to 0.001)')
    parser.add_argument('--gen_beta_1', type = float, default = 0.5, help = 'Generators optimizer adam beta one (Defaults to 0.5)')
    parser.add_argument('--gen_beta_2', type = float, default = 0.9, help = 'Generators optimizer adam beta two (Defaults to 0.9)')
    parser.add_argument('--gen_steps', type = int, default = 5, help = 'Training steps the generator takes per batch (Defaults to 5)')
    parser.add_argument('--disc_lr', type = float, default = 1e-5, help = 'Discriminators learning rate (Defaults to 0.0001)')
    parser.add_argument('--disc_beta_1', type = float, default = 0.5, help = 'Discriminators adam beta one (Defaults to 0.5)')
    parser.add_argument('--disc_beta_2', type = float, default = 0.9, help = 'Discriminators dam beta two (Defaults to 0.9)')
    parser.add_argument('--disc_steps', type = int, default = 1, help = 'Steps the discriminator takes per batch (Defaults to 1)')
    parser.add_argument('--latent_dim', type = float, default = 10.0, help = 'Latent dimension hyperparameter (Defaults to 10.0)')
    parser.add_argument('--new', type = bool, default = False, help = 'Whether to rebuild model from scratch (defaults to False)')
    parser.add_argument('--xla', type = bool, default = False, help = 'Whether to use accelerated linear algebra (XLA) (defaults to False)')

    # Parse the arguments
    args = parser.parse_args()

    configure_device(args)

    gen_config, disc_config = configure_model(args)

    if args.mode == "train":
        dataset = load_dataset(gen_config)

        generator = Generator(gen_config)
        discriminator = Discriminator(disc_config)

        trainer = Trainer(generator, discriminator, dataset)
        trainer.train()
    
    if args.mode == "generate":

        generate()

if __name__ == "__main__":

    main()

