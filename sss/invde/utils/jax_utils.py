import jax
import jax.numpy as jnp
import numpy as onp
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Callable
from scipy import ndimage

import ceviche_challenges as cc
from ceviche_challenges import units as u

from dataclasses import dataclass

import gin
import functools

@gin.configurable
def _get_default_initializer(
    mean=0.5,
    stddev=0.1,
    length_scale=1.0,
    bounds=(0.0, 1.0)
):
    print("mean, stddev, length_scale, bounds: ", mean, stddev, length_scale, bounds)
    return functools.partial(  # pyre-ignore[5]
                grayscale_initializer,
                mean=mean,
                stddev=stddev,
                length_scale=length_scale,
                bounds=bounds,
           )

def box_downsample(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    """Downsamples `x` to a coarser resolution array using box downsampling.

    Box downsampling forms nonoverlapping windows and simply averages the
    pixels within each window. For example, downsampling `(0, 1, 2, 3, 4, 5)`
    with a factor of `2` yields `(0.5, 2.5, 4.5)`.

    This function is a no-op when `factor` is `1`.

    Args:
        x: The array to be converted.
        factor: The factor determining the output shape.

    Returns:
        The output array, having shape `(a // factor, b // factor, ...)` if `x`
        has shape `(a, b, ...)`.
    """
    if factor == 1:
        return x

    if any([(d % factor) != 0 for d in x.shape]):
        raise ValueError(
            f"`factor` multiplying the axis sizes of `x` must be integers, "
            f"but got {factor} when `x` has shape {x.shape}."
        )
    shape = sum([(d // factor, factor) for d in x.shape], ())  # pyre-ignore[6]
    axes = list(range(1, 2 * x.ndim, 2))
    x = x.reshape(shape)
    return jnp.mean(x, axis=axes)

def kronecker_upsample(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    """Upsamples `x` to a finer resolution array using the Kronecker product.

    Kronecker product upsampling simply replicates pixel values, e.g.
    upsampling `(0, 1, 2)` by a factor of `2` yields `(0, 0, 1, 1, 2, 2)`.

    This function is a no-op when `factor` is `1`.

    Args:
        x: The array to be converted.
        factor: The integer factor determining the output shape.

    Returns:
        The output array, having shape `(a * factor, b * factor, ...)` if `x`
        has shape `(a, b, ...)`.
    """
    if factor == 1:
        return x

    kernel = jnp.ones((factor,) * x.ndim, dtype=x.dtype)
    return jnp.kron(x, kernel)

def wavelengths_from_wavelengths_nm(
    wavelengths_nm: Union[onp.ndarray, Sequence[float]]
) -> Sequence[u.Quantity]:
    """Converts wavelengths (in nanometers) to a sequence with units."""
    return tuple([wvl * u.nm for wvl in wavelengths_nm])


def sim_params(
    resolution_nm: int,
    wavelengths_nm: Union[onp.ndarray, Sequence[float]],
) -> cc.params.CevicheSimParams:
    """Convenience function to generate `CevicheSimParams` from unitless arguments."""
    return cc.params.CevicheSimParams(
        resolution=resolution_nm * u.nm,
        wavelengths=wavelengths_from_wavelengths_nm(wavelengths_nm),
    )


def extract_fixed_solid_void_pixels(
    density_with_gray_design: onp.ndarray,
) -> Tuple[onp.ndarray, onp.ndarray]:
    """Returns the fixed solid and void pixels.

    In `density_with_gray_design`, gray pixels are those not in `{0, 1}`.
    The minimal bounding box containing all gray pixels treated as the
    design. In this bounding box, the interior border pixels are set to be
    either fixed void or fixed solid, depending upon the neighboring pixel
    outside of the design, with "horizontal" neighbors taking precedence over
    "vertical" neighbors for corner pixels.

    Args:
        density_with_gray_design: Rank-2 array whose elements not in `{0, 1}`
            are interpreted as the design. Must not have any grayscale values
            on the border.

    Returns:
       The fixed solid and fixed void pixels, with shape equal to the design
       bounding box.
    """
    void = density_with_gray_design == 0
    solid = density_with_gray_design == 1
    design_mask = ~solid & ~void
    i, j = onp.nonzero(design_mask)
    i_min, j_min = i[0], j[0]
    i_max, j_max = i[-1], j[-1]

    assert i_min > 0
    assert j_min > 0
    assert i_max < solid.shape[0] - 1
    assert j_max < solid.shape[1] - 1

    fixed_solid = onp.zeros((i_max - i_min + 1, j_max - j_min + 1), dtype=bool)
    fixed_void = onp.zeros((i_max - i_min + 1, j_max - j_min + 1), dtype=bool)

    for fixed_arr, arr in [(fixed_solid, solid), (fixed_void, void)]:
        fixed_arr[:, 0] = arr[i_min : i_max + 1, j_min - 1]
        fixed_arr[:, -1] = arr[i_min : i_max + 1, j_max + 1]
        fixed_arr[0, :] = arr[i_min - 1, j_min : j_max + 1]
        fixed_arr[-1, :] = arr[i_max + 1, j_min : j_max + 1]

    return fixed_solid, fixed_void

@dataclass
class ResponseArray:
    """Stores the response of a component to excitation.

    The component has input and output ports, with each excitation being for a
    specific port and wavelength. The response to the excitation consists of a
    quantity for each output port. Thus, the response generally has the shape,
    `(..., num_wavelengths, num_excite_ports, num_output_ports)`.

    A specific example of a response are the scattering parameters.

    Attributes:
        array: The array.
        wavelengths_nm: The wavelengths for the scattering parameters.
        excite_port_idxs: The input port indices.
        output_port_idxs: The output port indices.
        shape: The shape of the array.
    """

    array: jnp.ndarray
    wavelengths_nm: Sequence[float]
    excite_port_idxs: Sequence[int]
    output_port_idxs: Sequence[int]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    def __post_init__(self) -> None:
        expected_shape = (
            len(self.wavelengths_nm),
            len(self.excite_port_idxs),
            len(self.output_port_idxs),
        )
        if self.array.shape[-3:] != expected_shape:
            raise ValueError(
                f"`array` has incompatible shape, expected trailing axes to "
                f"have shape {expected_shape} but got {self.array.shape}."
            )


# Register the `ResponseArray` dataclass, so that jax can manipulate it.
jax.tree_util.register_pytree_node(
    nodetype=ResponseArray,
    flatten_func=lambda s: (
        (s.array,),
        (s.wavelengths_nm, s.excite_port_idxs, s.output_port_idxs),
    ),
    unflatten_func=lambda context, array: ResponseArray(*array, *context),
)


@dataclass
class Density:
    """Stores a density array, and fixed pixels.

    Fixed pixels are those which are intended to be solid or intended to be
    void, i.e. `1`- or `0`-valued in a density.

    Note that this dataclass simply stores pixels that are to be fixed, it
    does not enforce these. When optimizing a density, a suitable algorithm
    should be used to ensure that fixed pixels have the intended values.

    Attributs:
        density: The density array.
        fixed_solid: Array indicating which pixels should be solid, or `None`.
        fixed_void: Array indicating which pixels should be void, or `None`.
        shape: The shape of the density array.
        model_output_dim: The output shape from the loaded generative model, if provided.
    """

    density: jnp.ndarray
    fixed_solid: Optional[jnp.ndarray] = None
    fixed_void: Optional[jnp.ndarray] = None
    model_output_dim: Optional[Tuple[int, ...]] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.density.shape

    def __post_init__(self) -> None:
        # Validate that fixed pixel shapes match `density`. When jax traces
        # a computation including `Density` objects, the `density` attribute
        # can be something other than an array. In this case, we avoid
        # validation that involes the `density`.
        if isinstance(self.density, jnp.ndarray):
            density_shape = (
                self.model_output_dim if self.model_output_dim else self.density.shape
            )
            if self.fixed_solid is not None and density_shape != self.fixed_solid.shape:
                raise ValueError(
                    f"`fixed_solid` must be `None` or match the shape of `density`, but "
                    f"got shape {self.fixed_solid.shape} when `density` has shape "
                    f"{density_shape}."
                )
            if self.fixed_void is not None and density_shape != self.fixed_void.shape:
                raise ValueError(
                    f"`fixed_void` must be `None` or match the shape of `density`, but "
                    f"got shape {self.fixed_void.shape} when `density` has shape "
                    f"{density_shape}."
                )
        # Validate that fixed pixels have `bool` type.
        if self.fixed_solid is not None and self.fixed_solid.dtype != bool:
            raise ValueError(
                f"`fixed_solid` must be `bool` but got type {self.fixed_solid.dtype}"
            )
        if self.fixed_void is not None and self.fixed_void.dtype != bool:
            raise ValueError(
                f"`fixed_void` must be `bool` but got type {self.fixed_void.dtype}"
            )


# Register the `Density` dataclass, so that jax can manipulate it.
jax.tree_util.register_pytree_node(
    nodetype=Density,
    flatten_func=lambda d: (
        (d.density,),
        (d.fixed_solid, d.fixed_void, d.model_output_dim),
    ),
    unflatten_func=lambda fixed, density: Density(*density, *fixed),
)

def transform_normal_array(
    normal_array: onp.ndarray,
    mean: float,
    stddev: float,
    length_scale: float,
    bounds: Optional[Tuple[float, float]],
) -> onp.ndarray:
    """Transforms an array to have given mean, standard deviation, and length scale.

    Shifting and scaling `normal_array` ensures that the array has the desired
    mean and standard deviation. The length scale in the output array is achieved
    by cropping and zooming appropriately.

    Note that the range of values in the output array may exceed those of the input.

    Args:
        random_normal_array: The array to be transformed; should have a mean
            of `0` and a standard deviation of `1`.
        mean: Desired mean value of the array.
        stddev: Desired stddev of the array.
        length_scale: Length scale of features in the array. Must be
            greater than `1`.
        bounds: Optional `(min, max)` giving the bounds to which values in the
            array are clipped.

    Returns:
        The transformed array.
    """
    if length_scale < 1:
        raise ValueError(f"`length_scale` must be at least `1` but got {length_scale}.")
    if bounds is not None and bounds[0] > bounds[1]:
        raise ValueError(f"`bounds` must be nondecreasing but got {bounds}.")

    shape = normal_array.shape

    # Crop the relevant region from `random_normal_array` before zooming.
    crop_shape = tuple([int(onp.ceil(s / length_scale)) for s in shape])
    array = normal_array[: crop_shape[0], : crop_shape[1]]
    array = array * stddev + mean

    # Zoom the array and crop, so that features have the required length scale.
    array = ndimage.zoom(array, length_scale)
    array = array[: shape[0], : shape[1]]

    if bounds is not None:
        array = onp.clip(array, *bounds)
    return array

def grayscale_initializer(
    key: jax.Array,
    shape: Tuple[int, int],
    mean: Union[float, Tuple[float, float]],
    stddev: float,
    length_scale: Union[float, Tuple[float, float]],
    bounds: Optional[Tuple[float, float]] = None,
    fixed_solid: Optional[jnp.ndarray] = None,
    fixed_void: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Returns a grayscale initial design array.

    Args:
        key: Used to generate the random array.
        shape: The desired design shape.
        mean: The desired mean value or a range from which the mean will be sampled.
        stddev: The desired standard deviation.
        length_scale: The desired length scale of features in the design or a range from
            which the length scale will be sampled.
        bounds: `(min, max)` values at which to clip the initial design.
        fixed_solid: Boolean array specifying which elements are to take
            the `max` value from `bounds`. If specified, `bounds` must
            not be `None`.
        fixed_void: Boolean array indicating which elements are to take
            the `min` value from `bounds`. Note that overlap of
            `fixed_solid` and `fixed_void` is not checked.

    Returns:
        The initial design array.
    """
    if fixed_solid is not None and bounds is None:
        raise ValueError("`bounds` must not be `None` if `fixed_solid` is given.")
    if fixed_solid is not None and fixed_solid.shape != shape:
        raise ValueError(
            f"`fixed_solid` shape must match `shape` but got {fixed_solid.shape} and {shape}"
        )
    if fixed_void is not None and bounds is None:
        raise ValueError("`bounds` must not be `None` if `fixed_void` is given.")
    if fixed_void is not None and fixed_void.shape != shape:
        raise ValueError(
            f"`fixed_void` shape must match `shape` but got {fixed_void.shape} and {shape}"
        )

    if type(mean) != float:
        rng, key = jax.random.split(key)
        mean = float(jax.random.uniform(rng, minval=mean[0], maxval=mean[1]))

    if type(length_scale) != float:
        rng, key = jax.random.split(key)
        length_scale = float(jax.random.uniform(rng, minval=5.0, maxval=10.0))

    density = transform_normal_array(
        normal_array=onp.asarray(jax.random.normal(key, shape)),
        mean=mean,
        stddev=stddev,
        length_scale=length_scale,
        bounds=bounds,
    )
    density = jnp.asarray(density)

    if fixed_void is not None and bounds is not None:
        density = jnp.where(fixed_void, jnp.full(shape, bounds[0]), density)
    if fixed_solid is not None and bounds is not None:
        density = jnp.where(fixed_solid, jnp.full(shape, bounds[1]), density)
    return density

def _rasterize_to_sim_grid(
    design: jnp.ndarray,
    design_resolution_nm: int,
    sim_resolution_nm: int,
) -> jnp.ndarray:
    """Rasterizes the design (at the design resolution) on the simulation grid."""
    if design_resolution_nm < sim_resolution_nm:
        factor = sim_resolution_nm // design_resolution_nm
        return box_downsample(design, factor)
    elif design_resolution_nm > sim_resolution_nm:
        factor = design_resolution_nm // sim_resolution_nm
        return kronecker_upsample(design, factor)
    else:
        return design

