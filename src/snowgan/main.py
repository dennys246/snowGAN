import os
from snowgan.trainer import Trainer
from snowgan.generate import generate
from snowgan.models.generator import load_generator
from snowgan.models.discriminator import load_discriminator
from snowgan.config import configure_generator, configure_discriminator
from snowgan.utils import parse_args, get_repo_root


def main():
    args = parse_args()

    if args.save_dir is None or os.path.exists(args.save_dir) is False:
        args.save_dir = str(get_repo_root()) + "/keras/snowgan/"

    if not os.path.exists(args.save_dir):
        print(f"Creating save directory: {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)

    gen_config = configure_generator(f"{args.save_dir}generator_config.json")
    disc_config = configure_discriminator(f"{args.save_dir}discriminator_config.json")

    # Load the generator
    generator = load_generator(f"{gen_config.save_dir}{gen_config.model_filename}")

    if args.mode == "train":
        # Load the discriminator
        discriminator = load_discriminator(f"{disc_config.save_dir}{disc_config.model_filename}")

        # Call to trainer
        trainer = Trainer(generator, discriminator)
        trainer.train(batch_size = args.batch_size, epochs = args.epochs)
    
    if args.mode == "generate":

        _ = generate(generator, args.synthetics, args.latent_dim, args.save_dir)

if __name__ == "__main__":

    main()

