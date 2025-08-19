import tensorflow as tf

def emd_loss(real_output, synthetic_output, gp, lambda_gp=10.0):
    """
    Wasserstein loss with gradient penalty for the Discriminator.

    Args:
        real_output (tf.Tensor): Discriminator output for real images
        synthetic_output (tf.Tensor): Discriminator output for fake images
        gp (tf.Tensor): Gradient penalty term
        lambda_gp (float): Weight for gradient penalty

    Returns:
        tf.Tensor: Total discriminator loss
    """
    wasserstein = tf.reduce_mean(synthetic_output) - tf.reduce_mean(real_output)
    return wasserstein + lambda_gp * gp

def generator_loss(synthetic_output):
    """
    Generator loss for Wasserstein GAN loss

    Args:
        synthetic_output (tf.Tensor): Discriminator output for fake images

    Returns:
        tf.Tensor: Generator loss
    """
    return -tf.reduce_mean(synthetic_output)

def compute_gradient_penalty(discriminator, real_images, fake_images, lambda_gp=10.0):
    """
    Computes gradient penalty for WGAN-GP.

    Args:
        discriminator (tf.keras.Model): The discriminator model
        real_images (tf.Tensor): Real images batch
        fake_images (tf.Tensor): Fake images batch
        lambda_gp (float): Penalty weight

    Returns:
        tf.Tensor: Gradient penalty scalar
    """
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
    penalty = tf.reduce_mean((norm - 1.0) ** 2) * lambda_gp
    return penalty
