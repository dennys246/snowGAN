import os, cv2, atexit, shutil
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class pipeline:

	def __init__(self, path, resolution):
		self.path = path
		self.profile_path = '/Users/dennyschaedig/Scripts/AvalancheAI/snow-profiles/magnified-profiles/'

		self.resolution = resolution

		self.images_loaded = self.read_loaded()
		if self.images_loaded:
			print(f"Previous images model trained on: {', '.join(self.images_loaded)}")
		else:
			print("No previously trained images detected...")

		self.avail_photos = [image for image in glob(f"{self.profile_path}*.JPG") if image not in self.images_loaded]
		print(f"Pipeline loaded with {len(self.avail_photos)} available photos to load...\n")
		

	def load_batch(self, count, resample = False):
		x = np.array([None])
		self.save_loaded()

		while self.avail_photos and x.shape[0] < count:
			image_filename = self.avail_photos.pop(0)
			self.images_loaded.append(image_filename)

			image = self.load_image(image_filename)
			if len(x.shape) > 1 and image.shape[0] == x.shape[1]:
				image = image.reshape((1, image.shape[0], image.shape[1], 3))
			else:
				image = image.reshape((1, image.shape[1], image.shape[0], 3))
			if len(x.shape) <= 1:
				x = image
			else:
				x = np.append(x, image, axis = 0)
			if resample:
				x = np.append(x, self.resample(image), axis = 0)
		return x


	# Load and preprocess images
	def load_image(self, path):
		print(f"Loading {path}")
		image = cv2.imread(path)  # Read image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, self.resolution)  # Resize
		image = (image / 127.5) - 1  # Normalize to -1, 1 for tanh
		image = np.array(image, dtype = np.float32)
		print(f"Image shape {image.shape} | max {image.max()} | deviation {image.std()}")
		return image

	def resample(self, image):
		synthetic_images = np.flip(image)
		return synthetic_images

	def save_loaded(self):
		# Remove the path if it exists
		if os.path.exists(f"{self.path}images_loaded.txt"):
			os.remove(f"{self.path}images_loaded.txt")

		# Open a new file and output the images trained on
		with open(f"{self.path}images_loaded.txt", 'w') as file:
			for image in self.images_loaded:
				file.write(image+'\n')

	def read_loaded(self):
		if os.path.exists(f"{self.path}images_loaded.txt"):
			with open(f"{self.path}images_loaded.txt", 'r') as file:
				return [line.split('\n')[0] for line in file.readlines()]
		else: # If no saved record of trained images exists
			return [] # Return an empty list
