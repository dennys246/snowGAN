"""
Differentiable augmentation for GAN training.

All operations are differentiable so gradients flow through augmentation
back to the generator. Applied to both real and fake images before the
discriminator to prevent discriminator overfitting.

Reference: Zhao et al., "Differentiable Augmentation for Data-Efficient GAN Training" (2020)
"""

import tensorflow as tf


def augment(images, p=0.5):
    """
    Apply random differentiable augmentations to a batch of images.

    Args:
        images: (B, H, W, C) tensor in [-1, 1] range
        p: probability of applying each augmentation

    Returns:
        Augmented images tensor, same shape and range
    """
    images = _random_flip(images, p)
    images = _random_brightness(images, p, max_delta=0.2)
    images = _random_saturation(images, p, lower=0.8, upper=1.2)
    images = _random_cutout(images, p, ratio=0.25)
    return images


def _random_flip(images, p):
    if tf.random.uniform([]) < p:
        images = tf.image.random_flip_left_right(images)
    return images


def _random_brightness(images, p, max_delta=0.2):
    if tf.random.uniform([]) < p:
        delta = tf.random.uniform([], -max_delta, max_delta)
        images = images + delta
        images = tf.clip_by_value(images, -1.0, 1.0)
    return images


def _random_saturation(images, p, lower=0.8, upper=1.2):
    if tf.random.uniform([]) < p:
        # Convert from [-1,1] to [0,1] for saturation adjustment
        images_01 = (images + 1.0) / 2.0
        factor = tf.random.uniform([], lower, upper)
        mean = tf.reduce_mean(images_01, axis=-1, keepdims=True)
        images_01 = mean + factor * (images_01 - mean)
        images = tf.clip_by_value(images_01 * 2.0 - 1.0, -1.0, 1.0)
    return images


def _random_cutout(images, p, ratio=0.25):
    """Random rectangular cutout filled with zeros."""
    if tf.random.uniform([]) < p:
        shape = tf.shape(images)
        batch_size = shape[0]
        h = shape[1]
        w = shape[2]
        cut_h = tf.cast(tf.cast(h, tf.float32) * ratio, tf.int32)
        cut_w = tf.cast(tf.cast(w, tf.float32) * ratio, tf.int32)

        # Random top-left corner for each image in the batch
        cy = tf.random.uniform([batch_size, 1, 1, 1], 0, tf.cast(h, tf.float32), dtype=tf.float32)
        cx = tf.random.uniform([batch_size, 1, 1, 1], 0, tf.cast(w, tf.float32), dtype=tf.float32)

        # Create coordinate grids
        y_range = tf.cast(tf.reshape(tf.range(h), [1, h, 1, 1]), tf.float32)
        x_range = tf.cast(tf.reshape(tf.range(w), [1, 1, w, 1]), tf.float32)

        # Mask: 0 inside cutout, 1 outside
        mask_y = tf.cast((y_range >= cy) & (y_range < cy + tf.cast(cut_h, tf.float32)), tf.float32)
        mask_x = tf.cast((x_range >= cx) & (x_range < cx + tf.cast(cut_w, tf.float32)), tf.float32)
        mask = 1.0 - mask_y * mask_x

        images = images * mask
    return images
