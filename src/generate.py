import os, cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from PIL import Image

def generate(generator, count = 1, seed_size = 100, save_dir = None, filename_prefix = 'synthetic'):
    """
    Generate synthetic images using the currently loaded generator and save
    the images to the model's synthetic images folder.

    Function arguments:
        model (keras model) - GAN model to generate synthetic images from
        count (int) - Number of synthetic images to create
        seed_size (size) - Size of the input noise seed to the generator
        save_dir (str) - Folder path to save images, if none provided images are
        filename_prefix (str) - prefix to give the filename
    """

    # Check if the destimation exists
    previously_generated = 0
    if os.path.exists(save_dir): # If folder exists
        # Check for previously generated images
        previous_images = glob(f"{save_dir}{filename_prefix}*.png")
    else: # If not
        print(f"Output folder doesn't exist, creating directory")
        os.makedirs(f"{save_dir}", exist_ok = True) # Create folder

    # Create a random fix seed for consistent visualization
    seed = tf.random.normal([count, seed_size])

    # Generate synthetic images
    synthetic_images = generator(seed, training = False)
    print(f"Synthetic images generated: {synthetic_images.shape}")
    
    # Iterate through each synthetic image
    for ind in range(synthetic_images.shape[0]):
        
        # Construct image filename
        filepath = f"{save_dir}{filename_prefix}_{previously_generated + ind + 1}.png"

        # Reformat generated image to RGB from BGR
        image_arr = synthetic_images[ind].numpy()
        print(f"Image (prior) shape {image_arr.shape} | max {image_arr.max()} | deviation {image_arr.std()}")
        image_arr = np.clip((image_arr + 1) * 127.5, 0, 255).astype(np.uint8)
        print(f"Image (prior) shape {image_arr.shape} | max {image_arr.max()} | deviation {image_arr.std()}")

        # Convert to PIL image
        image = Image.fromarray(image_arr)

        # Save image with parameters provided
        save_image(image, filepath)
    
    return synthetic_images

def save_image(image, filepath):
    # Ensure output dir exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert to image and save
    image.save(filepath)
    print(f"Saved image to {filepath}")

def make_movie(save_dir = "outputs", videoname = "snowgan_synthetics.mp4", framerate = 15, filepath_pattern = "*.png"):
    """
    Create a .mp4 movie of the batch history of synthetic images generated to 
    display the progression of what features the snowGAN generator (and presumably 
    discriminator) learned.

    Function arguments:
        folder (str) - Folderpath where synthetic images to be made into movie are stored
        videoname (str) - String of the videoname to save the .mp4 file generated
        framerate (int) - Framerate to set the .mp4 video of synthetic image history
        filepath_pattern (str) - File path pattern to glob synthetic images with
    """
    
    videoname = f"{save_dir}{videoname}" #Define video name using path and video
    
    # Grab all synthetic images
    synthetic_files = sorted(glob(f"{save_dir}/{filepath_pattern}")) # Grab all synthetic images

    # Grab each synthetic image
    synthetic_numbers = [int(file.split('.')[0].split('_')[-1]) for file in synthetic_files]

    # Sort synthetic files in order
    zipper = zip(synthetic_numbers, synthetic_files)
    zipper = sorted(zipper)
    synthetic_numbers, synthetic_files = zip(*zipper)
    
    # Read the first image to get dimensions
    image = cv2.imread(synthetic_files[0])
    height, width, layers = image.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    video = cv2.VideoWriter(videoname, fourcc, framerate, (width, height))

    # Add images to the video
    for image_file in synthetic_files:
        image = cv2.imread(image_file)
        video.write(image)

    # Release the video writer and clear memory
    video.release()
    cv2.destroyAllWindows()
