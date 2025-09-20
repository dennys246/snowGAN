import os

def save_history(save_dir, loss, trained_data):
    """
    Plot and save the generator and discriminator history loaded in the snowGAN object
    """
    # Make directories that don't exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the current generate loss progress
    with open(f"{save_dir}generator_loss.txt", "w") as file:
        for loss_datum in loss['gen']:
            file.write(f"{loss_datum}\n")

    # Save the current discriminator loss
    with open(f"{save_dir}discriminator_loss.txt", "w") as file:
        for loss_datum in loss['disc']:
            file.write(f"{loss_datum}\n")

    with open(f"{save_dir}trained.txt", "w") as file:
        for trained in trained_data:
            file.write(f"{trained}\n")

def load_history(save_dir):
    """
    Load generator/discriminator loss history and trained data from text files
    and assign them to the snowGAN object.
    """

    gen_path = os.path.join(save_dir, "generator_loss.txt")
    disc_path = os.path.join(save_dir, "discriminator_loss.txt")
    trained_path = os.path.join(save_dir, "trained.txt")

    # Initialize containers
    loss = {"gen": [], "disc": []}
    trained_data = []

    # Load generator loss
    if os.path.exists(gen_path):
        with open(gen_path, "r") as file:
            loss["gen"] = [float(line.strip()) for line in file if line.strip()]
    
    # Load discriminator loss
    if os.path.exists(disc_path):
        with open(disc_path, "r") as file:
            loss["disc"] = [float(line.strip()) for line in file if line.strip()]

    # Load trained data
    if os.path.exists(trained_path):
        with open(trained_path, "r") as file:
            trained_data = [line.strip() for line in file if line.strip()]
        
    return loss, trained_data

 