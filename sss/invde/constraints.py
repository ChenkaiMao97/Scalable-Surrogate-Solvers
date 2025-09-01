import gin
from typing import Callable, Tuple, List

import jax
import jax.numpy as jnp

def concatenate_jax_arrays(arrays: List[jnp.ndarray]) -> jnp.ndarray:
    flattened_arrays = [jnp.reshape(arr, (-1,)) for arr in arrays]
    concatenated_array = jnp.concatenate(flattened_arrays)
    return concatenated_array

def segment_jax_vector(
    concatenated_vector: jnp.ndarray, num_arrays: int, array_shape: Tuple[int]
) -> List[jnp.ndarray]:
    segments = jnp.split(concatenated_vector, num_arrays)
    reshaped_arrays = [jnp.reshape(segment, array_shape) for segment in segments]
    return reshaped_arrays

def indicator_solid(
    x: jnp.array,
    c: float,
    filter_f: Callable,
    threshold_f: Callable,
) -> float:
    filtered_field = filter_f(x)
    design_field = threshold_f(filtered_field)
    gradient_filtered_field = jnp.gradient(filtered_field)
    grad_mag = (gradient_filtered_field[0]) ** 2 + (gradient_filtered_field[1]) ** 2
    assert grad_mag.shape == x.shape, "grad shape has changed."
    return design_field * jnp.exp(-c * grad_mag)

def indicator_void(
    x: jnp.array,
    c: float,
    filter_f: Callable,
    threshold_f: Callable,
) -> float:
    filtered_field = filter_f(x)
    design_field = threshold_f(filtered_field)
    gradient_filtered_field = jnp.gradient(filtered_field)
    grad_mag = (gradient_filtered_field[0]) ** 2 + (gradient_filtered_field[1]) ** 2
    assert grad_mag.shape == x.shape, "grad shape has changed."
    return (1 - design_field) * jnp.exp(-c * grad_mag)

def constraint_solid(
    x: jnp.array,
    c: float,
    eta_e: float,
    filter_f: Callable,
    threshold_f: Callable,
):
    filtered_field = filter_f(x)
    I_s = indicator_solid(
        x.reshape(filtered_field.shape),
        c,
        filter_f,
        threshold_f,
    ).flatten()
    return jnp.mean(I_s * jnp.minimum(filtered_field.flatten() - eta_e, 0) ** 2)

def constraint_void(
    x: jnp.array,
    c: float,
    eta_d: float,
    filter_f: Callable,
    threshold_f: Callable,
) -> float:
    filtered_field = filter_f(x)
    I_v = indicator_void(
        x.reshape(filtered_field.shape),
        c,
        filter_f,
        threshold_f,
    ).flatten()
    return jnp.mean(I_v * jnp.minimum(eta_d - filtered_field.flatten(), 0) ** 2)

@gin.configurable
def constraint_solid_decode(
    x: jnp.ndarray,
    decode_fn: Callable,
    threshold_fn: Callable,
    filter_fn: Callable,
    indicator_c: jnp.ndarray,
    eta_e: float,
    constraint_tolerance: float,
):
    x = decode_fn(x)
    return (
        jnp.log10(
            constraint_solid(
                x,
                indicator_c,
                eta_e,
                filter_fn,
                threshold_fn,
            )
            + 1e-12
        )
        + constraint_tolerance
    )

@gin.configurable
def constraint_void_decode(
    x: jnp.ndarray,
    decode_fn: Callable,
    threshold_fn: Callable,
    filter_fn: Callable,
    indicator_c: jnp.ndarray,
    eta_d: float,
    constraint_tolerance: float,
):
    x = decode_fn(x)
    return (
        jnp.log10(
            constraint_void(
                x,
                indicator_c,
                eta_d,
                filter_fn,
                threshold_fn,
            )
            + 1e-12
        )
        + constraint_tolerance
    )

@gin.configurable
def constraints_function(
    beta: float,
    length_scale_pixel: int,
    projection_fn: Callable,
    solid_contraints_fn: Callable,
    void_contraints_fn: Callable,
    filter_fn: Callable,
    decode_fn: Callable,
    latent_shape: Tuple[int, int],
    constraint_tolerance: float,
    num_layers: int = 1,
):
    conic_filter_fn = lambda density: filter_fn(
        density, radius=length_scale_pixel, is_periodic=True
    )
    tanh_fn = lambda density: projection_fn(density, beta=beta)

    def c_solid(x: jnp.ndarray):
        return solid_contraints_fn(
            x,
            decode_fn=decode_fn,
            threshold_fn=tanh_fn,
            filter_fn=conic_filter_fn,
            constraint_tolerance=constraint_tolerance,
        )

    def c_void(x: jnp.ndarray):
        return void_contraints_fn(
            x,
            decode_fn=decode_fn,
            threshold_fn=tanh_fn,
            filter_fn=conic_filter_fn,
            constraint_tolerance=constraint_tolerance,
        )

    def _vector_valued_constraint_function(
        result: jnp.ndarray,
        x: jnp.ndarray,
        grad: jnp.ndarray,
    ) -> None:
        """An nlopt-friendly, vector-valued function for the manufacturing constraints."""

        layers = segment_jax_vector(x, num_layers, latent_shape)
        # The new constraint functions are _aggregates_ of each layer's individual
        # constraint function. A simple sum.
        layer_solid, layer_void = [], []
        aggregate_solid_value, aggregate_void_value = 0, 0
        for layer in layers:
            c_solid_value_and_grad_fn = jax.value_and_grad(c_solid)
            c_void_value_and_grad_fn = jax.value_and_grad(c_void)

            c_solid_val, c_solid_grad = c_solid_value_and_grad_fn(layer)
            c_void_val, c_void_grad = c_void_value_and_grad_fn(layer)

            aggregate_solid_value += c_solid_val
            aggregate_void_value += c_void_val

            layer_solid.append(c_solid_grad.flatten())
            layer_void.append(c_void_grad.flatten())

        result[0] = float(aggregate_solid_value)
        result[1] = float(aggregate_void_value)
        print("constraints: ", result)
        grad_solid = concatenate_jax_arrays(layer_solid)
        grad_void = concatenate_jax_arrays(layer_void)
        grad[0, :] = (
            jnp.zeros_like(grad_solid)
            if float(aggregate_solid_value) <= 0
            else grad_solid
        )
        grad[1, :] = (
            jnp.zeros_like(grad_void) if float(aggregate_void_value) <= 0 else grad_void
        )
    
    def constraint_value_and_grad(
        x: jnp.ndarray,
    ):
        """An nlopt-friendly, vector-valued function for the manufacturing constraints."""

        layers = segment_jax_vector(x, num_layers, latent_shape)
        # The new constraint functions are _aggregates_ of each layer's individual
        # constraint function. A simple sum.
        layer_solid, layer_void = [], []
        aggregate_solid_value, aggregate_void_value = 0, 0
        for layer in layers:
            c_solid_value_and_grad_fn = jax.value_and_grad(c_solid)
            c_void_value_and_grad_fn = jax.value_and_grad(c_void)

            c_solid_val, c_solid_grad = c_solid_value_and_grad_fn(layer)
            c_void_val, c_void_grad = c_void_value_and_grad_fn(layer)

            aggregate_solid_value += c_solid_val
            aggregate_void_value += c_void_val

            layer_solid.append(c_solid_grad.flatten())
            layer_void.append(c_void_grad.flatten())

        grad_solid = concatenate_jax_arrays(layer_solid)
        grad_void = concatenate_jax_arrays(layer_void)

        return aggregate_solid_value, aggregate_void_value, grad_solid.reshape(latent_shape), grad_void.reshape(latent_shape)

    return _vector_valued_constraint_function, constraint_value_and_grad