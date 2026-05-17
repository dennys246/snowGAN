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

    Accepts ``(B, H, W, C)`` (legacy 2-D path) or ``(B, D, H, W, C)`` (depth-axis
    modality contract). For 5-D inputs, the same flip / cutout decision is
    applied across the depth axis so that paired modalities in merged mode stay
    coherent.

    Args:
        images: image tensor in [-1, 1] range
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
        if len(images.shape) == 5:
            # tf.image.random_flip_left_right only handles 3-D / 4-D. Roll our
            # own 5-D flip with one decision per batch element broadcast across
            # the depth axis so paired modalities flip together.
            batch = tf.shape(images)[0]
            flip = tf.random.uniform([batch]) > 0.5
            flip = tf.reshape(flip, [batch, 1, 1, 1, 1])
            flipped = tf.reverse(images, axis=[3])  # width axis
            images = tf.where(flip, flipped, images)
        else:
            images = tf.image.random_flip_left_right(images)
    return images


def _random_brightness(images, p, max_delta=0.2):
    if tf.random.uniform([]) < p:
        delta = tf.random.uniform([], -max_delta, max_delta, dtype=images.dtype)
        images = images + delta
        images = tf.clip_by_value(images, tf.cast(-1.0, images.dtype), tf.cast(1.0, images.dtype))
    return images


def _random_saturation(images, p, lower=0.8, upper=1.2):
    if tf.random.uniform([]) < p:
        # Convert from [-1,1] to [0,1] for saturation adjustment
        images_01 = (images + tf.cast(1.0, images.dtype)) / tf.cast(2.0, images.dtype)
        factor = tf.random.uniform([], lower, upper, dtype=images.dtype)
        mean = tf.reduce_mean(images_01, axis=-1, keepdims=True)
        images_01 = mean + factor * (images_01 - mean)
        images = tf.clip_by_value(images_01 * tf.cast(2.0, images.dtype) - tf.cast(1.0, images.dtype),
                                  tf.cast(-1.0, images.dtype), tf.cast(1.0, images.dtype))
    return images


def _random_cutout(images, p, ratio=0.25):
    """Random rectangular cutout filled with zeros.

    Supports ``(B, H, W, C)`` and ``(B, D, H, W, C)``; the same cutout
    rectangle is applied to every depth slice of a given batch element.
    """
    if tf.random.uniform([]) < p:
        rank = len(images.shape)
        shape = tf.shape(images)
        batch_size = shape[0]
        if rank == 5:
            h = shape[2]
            w = shape[3]
            mask_shape_y = [1, 1, h, 1, 1]
            mask_shape_x = [1, 1, 1, w, 1]
            cy_shape = [batch_size, 1, 1, 1, 1]
            cx_shape = [batch_size, 1, 1, 1, 1]
        else:
            h = shape[1]
            w = shape[2]
            mask_shape_y = [1, h, 1, 1]
            mask_shape_x = [1, 1, w, 1]
            cy_shape = [batch_size, 1, 1, 1]
            cx_shape = [batch_size, 1, 1, 1]

        cut_h = tf.cast(tf.cast(h, tf.float32) * ratio, tf.int32)
        cut_w = tf.cast(tf.cast(w, tf.float32) * ratio, tf.int32)

        cy = tf.random.uniform(cy_shape, 0, tf.cast(h, tf.float32), dtype=tf.float32)
        cx = tf.random.uniform(cx_shape, 0, tf.cast(w, tf.float32), dtype=tf.float32)

        y_range = tf.cast(tf.reshape(tf.range(h), mask_shape_y), tf.float32)
        x_range = tf.cast(tf.reshape(tf.range(w), mask_shape_x), tf.float32)

        mask_y = tf.cast((y_range >= cy) & (y_range < cy + tf.cast(cut_h, tf.float32)), tf.float32)
        mask_x = tf.cast((x_range >= cx) & (x_range < cx + tf.cast(cut_w, tf.float32)), tf.float32)
        mask = 1.0 - mask_y * mask_x

        images = images * tf.cast(mask, images.dtype)
    return images
