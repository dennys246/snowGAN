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


def _per_image_shape(images):
    """Broadcast shape ``[B, 1, ..., 1]`` for one parameter/decision per image.

    Every augmentation below draws its apply-decision and its magnitude at this
    shape so each image in the batch is augmented independently. The previous
    implementation drew a single scalar per *batch* (one coin flip, one delta
    for all images), which collapsed augmentation entropy — on a batch_size=4
    run roughly half the steps applied nothing at all, leaving ADA almost no
    overfitting signal to act on (DiffAugment expects per-image transforms).
    """
    rank = len(images.shape)
    batch = tf.shape(images)[0]
    return tf.concat([[batch], tf.ones([rank - 1], dtype=tf.int32)], axis=0)


def _random_flip(images, p):
    # One flip decision per image, broadcast across depth so paired modalities
    # (5-D) flip together. Width is axis 3 for (B,D,H,W,C), axis 2 for (B,H,W,C).
    shape = _per_image_shape(images)
    flip = tf.random.uniform(shape) < p
    width_axis = 3 if len(images.shape) == 5 else 2
    flipped = tf.reverse(images, axis=[width_axis])
    return tf.where(flip, flipped, images)


def _random_brightness(images, p, max_delta=0.2):
    shape = _per_image_shape(images)
    apply = tf.cast(tf.random.uniform(shape) < p, images.dtype)
    delta = tf.random.uniform(shape, -max_delta, max_delta, dtype=images.dtype)
    images = images + apply * delta
    return tf.clip_by_value(images, tf.cast(-1.0, images.dtype), tf.cast(1.0, images.dtype))


def _random_saturation(images, p, lower=0.8, upper=1.2):
    shape = _per_image_shape(images)
    apply = tf.cast(tf.random.uniform(shape) < p, images.dtype)
    one = tf.cast(1.0, images.dtype)
    # factor == 1 (identity) for images that don't get the augment this step.
    factor = one + apply * (tf.random.uniform(shape, lower, upper, dtype=images.dtype) - one)
    images_01 = (images + one) / tf.cast(2.0, images.dtype)
    mean = tf.reduce_mean(images_01, axis=-1, keepdims=True)
    images_01 = mean + factor * (images_01 - mean)
    return tf.clip_by_value(images_01 * tf.cast(2.0, images.dtype) - one,
                            tf.cast(-1.0, images.dtype), one)


def _random_cutout(images, p, ratio=0.25):
    """Random rectangular cutout filled with zeros, decided per image.

    Supports ``(B, H, W, C)`` and ``(B, D, H, W, C)``; the cutout rectangle is
    applied to every depth slice of a given batch element, but the rectangle
    position AND whether it fires are now drawn independently per image.
    """
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
    keep = 1.0 - mask_y * mask_x  # 0 inside the cut rectangle, 1 elsewhere

    # Per-image apply gate: images that don't fire keep everything (gate via the
    # cut amount, so keep_eff == 1 where apply == 0).
    apply = tf.cast(tf.random.uniform(cy_shape) < p, tf.float32)
    keep_eff = 1.0 - apply * (1.0 - keep)
    return images * tf.cast(keep_eff, images.dtype)
