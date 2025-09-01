import gin
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt

def build_kernel_grid(x, x_shape):
    n,m = x_shape
    xv = jnp.arange(n//2 + 1)
    yv = jnp.arange(m//2 + 1)
    xv, yv = jnp.meshgrid(xv, yv, indexing='ij', sparse=True)
    return xv, yv

@gin.configurable
def conic_filter(x, radius, is_periodic=False):
    X, Y = build_kernel_grid(x, x.shape)

    kernel = jnp.where(jnp.abs(X**2 + Y**2) <= radius**2, (1 - jnp.sqrt(jnp.abs(X**2 + Y**2)) / radius), 0)
    return padded_conv_fft(x, kernel, padding_mode='wrap' if is_periodic else 'edge')

def _build_kernel_from_quadrant(kernel, kernel_shape, pad_to):
    """Build the full kernel from the nonegative quadrant."""

    pad_size = (
        pad_to[0] - 2 * kernel_shape[0] + 1,
        pad_to[1] - 2 * kernel_shape[1] + 1,
    )

    top = jnp.zeros((pad_size[0], kernel_shape[1]))
    bottom = jnp.zeros((pad_size[0], kernel_shape[1] - 1))
    middle = jnp.zeros((pad_to[0], pad_size[1]))

    top_left_corner = kernel[:, :]
    top_right_corner = jnp.flipud(kernel[1:, :])
    bot_left_corner = jnp.fliplr(kernel[:, 1:])
    bot_right_corner = jnp.flip(kernel[1:, 1:])

    return jnp.concatenate(
        (
            jnp.concatenate((top_left_corner, top, top_right_corner)),
            middle,
            jnp.concatenate((bot_left_corner, bottom, bot_right_corner)),
        ),
        axis=1,
    )

def padded_conv_fft(arr, kernel, padding_mode='edge'):
    assert kernel.ndim == 2

    shape = arr.shape[-2:]
    pad = max(shape)

    batch_dims = arr.ndim - 2

    padding = [(0, 0)] * batch_dims + [(pad, pad), (pad, pad)]
    arr = jnp.pad(arr, padding, mode=padding_mode)

    kernel = _build_kernel_from_quadrant(kernel, kernel.shape, arr.shape[-2:])
    kernel = kernel / jnp.sum(kernel)  # normalize
    y = jnp.fft.ifft2(jnp.fft.fft2(arr) * jnp.fft.fft2(kernel))
    i_lo = pad
    j_lo = pad
    i_hi = shape[0] + i_lo
    j_hi = shape[1] + j_lo
    y = y[..., i_lo:i_hi, j_lo:j_hi]
    return y.astype(jnp.promote_types(arr.dtype, kernel.dtype))

@gin.configurable
def tanh_projection(x, beta, eta=0.5):
    if beta == jnp.inf:
        return jnp.where(x > eta, 1.0, 0.0)
    if beta == 0:
        return x
    if eta == 0:
        return jnp.zeros_like(x)
    if eta == 1:
        return jnp.ones_like(x)
    return jnp.where(
        ((x == eta) & (beta == jnp.inf)),
        0.5,
        (jnp.tanh(beta * eta) + jnp.tanh(beta * (x - eta)))
        / (jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta))),
    )

@gin.configurable
def identity_op(x):
    return x


if __name__ == '__main__':
    x = onp.random.rand(100, 100)
    y = conic_filter(x, 5)
    z = tanh_projection(y, 1e4)
    plt.subplot(1, 3, 1)
    plt.imshow(x)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(y)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(z)
    plt.colorbar()
    plt.savefig('conic_filter.png')