import tensorflow as tf


def compute_gradient_penalty(discriminator, real_images, fake_images):
    """
    Computes gradient penalty for WGAN-GP.

    Args:
        discriminator (tf.keras.Model): The discriminator model
        real_images (tf.Tensor): Real images batch
        fake_images (tf.Tensor): Fake images batch

    Returns:
        tf.Tensor: Gradient penalty scalar (unscaled — caller applies lambda_gp)
    """
    batch_size = tf.shape(real_images)[0]
    # Compute GP in float32 for numerical stability (important with mixed precision)
    real_f32 = tf.cast(real_images, tf.float32)
    fake_f32 = tf.cast(fake_images, tf.float32)
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    interpolated = alpha * real_f32 + (1 - alpha) * fake_f32

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
        pred = tf.cast(pred, tf.float32)

    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
    penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return penalty
