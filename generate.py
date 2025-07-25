import os 
import tensorflow as tf   

def generate(self, model, count = 1, seed_size = 100, folder = "./", filename_prefix = 'synthetic', title = "Synthetic Image", subfolder = None):
    """
    Generate synthetic images using the currently loaded generator and save
    the images to the model's synthetic images folder.

    Function arguments:
        model (keras model) - GAN model to generate synthetic images from
        count (int) - Number of synthetic images to create
        seed_size (size) - Size of the input noise seed to the generator
        folder (str) - Folder path where you would like to store generated images
        filename_prefix (str) - prefix to give the filename
        title (str) - Title to give the synthetically generated images
        subfolder (str) - Child folder to build and save synthetic images to
    """
    if subfolder: # If subfolder requested
        folder += subfolder # Add subfolder to path

    # Check if the destimation exists
    previously_generated = 0
    if os.path.exists(f"{folder}"): # If folder exists
        # Check for previously generated images
        previous_images = glob(f"{folder}{filename_prefix}*.png")
        previously_generated = len(previous_images)
    else: # If not
        print(f"Output folder doesn't exist, creating directory")
        os.makedirs(f"{folder}", exist_ok = True) # Create folder

    # Create a random fix seed for consistent visualization
    seed = tf.random.normal([count, ])

    # Generate synthetic images
    synthetic_images = model(seed, training=False)
    print(f"Synthetic images generated: {synthetic_images.shape}")
    
    # Iterate through each synthetic image
    for ind in range(synthetic_images.shape[0]):
        
        # Construct image filename
        filename = f"{folder}{filename_prefix}_{previously_generated + ind + 1}.png"

        # Reformat generated image to RGB from BGR
        image = synthetic_images[ind].numpy()
        print(f"Image (prior) shape {image.shape} | max {image.max()} | deviation {image.std()}")
        image = np.clip((image + 1) * 127.5, 0, 255).astype(np.uint8)
        print(f"Image (prior) shape {image.shape} | max {image.max()} | deviation {image.std()}")

        # Save image with parameters provided
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.savefig(filename)
        plt.close()
    return synthetic_images

def make_movie(self, folder, videoname = "snowgan_synthetics.mp4", framerate = 15, filepath_pattern = "*.png"):
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
    
    videoname = f"{folder}{videoname}" #Define video name using path and video
    
    # Grab all synthetic images
    synthetic_files = sorted(glob(f"{folder}/{filepath_pattern}")) # Grab all synthetic images

    # Grab each synthetic image
    synthetic_numbers = [int(file.split('.')[0].split('_')[-1]) for file in synthetic_files]

    # Sort synthetic files in order
    zipper = zip(synthetic_numbers, synthetic_files)
    zipper = sorted(zipper)
    synthetic_numbers, synthetic_files = zip(*zipper)
    
    # Read the first image to get dimensions
    image = cv2.imread(batches[0][0])
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
