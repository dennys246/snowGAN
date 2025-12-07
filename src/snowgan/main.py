import os
from snowgan.trainer import Trainer
from snowgan.generate import generate
from snowgan.models.generator import load_generator
from snowgan.models.discriminator import load_discriminator
from snowgan.config import configure_disc, configure_gen, build
from snowgan.inference import run_inference
from snowgan.utils import parse_args, configure_device


def main():
    args = parse_args()
    if args.mode == "infer":
        # Force CPU for inference to avoid GPU PTX compile requirements
        args.device = "cpu"
        args.xla = False
    # Configure TensorFlow runtime (GPU/CPU, XLA, mixed precision)
    try:
        configure_device(args)
    except Exception as e:
        print(f"Warning: device configuration failed: {e}")

    if not os.path.exists(args.save_dir):
        print(f"Creating save directory {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)

    gen_config_path = os.path.join(args.save_dir, "generator_config.json")
    gen_config = build(gen_config_path)
    gen_config = configure_gen(gen_config, args)
    # Persist configured generator settings (including fade/fade_steps) immediately
    gen_config.save_config(gen_config_path)

    disc_config_path = os.path.join(args.save_dir, "discriminator_config.json")
    disc_config = build(disc_config_path)
    disc_config = configure_disc(disc_config, args)
    # Persist configured discriminator settings immediately
    disc_config.save_config(disc_config_path)

    if args.mode == "infer":
        discriminator = load_discriminator(disc_config.checkpoint, disc_config)
        results = run_inference(
            discriminator,
            dataset_name=disc_config.dataset,
            resolution=disc_config.resolution,
            batch_size=args.batch_size,
            sample_count=args.infer_samples,
            save_dir=disc_config.save_dir,
        )
        print(f"Inference complete on {results['total_seen']} samples; plot saved to {results.get('plot_path')}")
        return

    # Load the generator when training or generating samples
    generator = load_generator(gen_config.checkpoint, gen_config)

    if args.mode == "train":
        # Load the discriminator
        discriminator = load_discriminator(disc_config.checkpoint, disc_config)

        # Call to trainer
        trainer = Trainer(generator, discriminator)
        trainer.train(batch_size = args.batch_size, epochs = args.epochs)
    
    elif args.mode == "generate":

        _ = generate(generator, args.n_samples, args.latent_dim, args.save_dir)

if __name__ == "__main__":

    main()

