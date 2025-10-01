import os
from snowgan.trainer import Trainer
from snowgan.generate import generate
from snowgan.models.generator import load_generator
from snowgan.models.discriminator import load_discriminator
from snowgan.config import load_disc_config, load_gen_config, configure_disc, configure_gen, config_template
from snowgan.utils import parse_args, get_repo_root


def main():
    args = parse_args()

    if args.save_dir is None or os.path.exists(args.save_dir) is False:
        args.save_dir = str(get_repo_root()) + "/keras/snowgan/"

    if not os.path.exists(args.save_dir):
        print(f"Creating save directory: {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)

    gen_config = config_template
    gen_config = configure_gen(gen_config, args)
    gen_config = load_gen_config(f"{args.save_dir}generator_config.json", gen_config)

    disc_config = config_template
    disc_config = configure_disc(disc_config, args)
    disc_config = load_disc_config(f"{args.save_dir}discriminator_config.json", disc_config)

    # Load the generator
    generator = load_generator(f"{gen_config.save_dir}{gen_config.checkpoint}")

    if args.mode == "train":
        # Load the discriminator
        discriminator = load_discriminator(f"{disc_config.save_dir}{disc_config.checkpoint}")

        # Call to trainer
        trainer = Trainer(generator, discriminator)
        trainer.train(batch_size = args.batch_size, epochs = args.epochs)
    
    if args.mode == "generate":

        _ = generate(generator, args.n_samples, args.latent_dim, args.save_dir)

if __name__ == "__main__":

    main()

