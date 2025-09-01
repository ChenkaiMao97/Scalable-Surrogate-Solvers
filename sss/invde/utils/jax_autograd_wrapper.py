"""Defines a jax-compatible wrapper for autograd functions."""

import dataclasses
from typing import Any, Callable, Tuple, Union

import autograd
import autograd.numpy as npa
import jax
import jax.numpy as jnp
import jax.tree_util as pytree
import numpy as onp

# Type alias for pytrees and arrays.
PyTree = Any  # pyre-ignore[33]
Value = Any  # pyre-ignore[33]


def jax_wrap_autograd(  # pyre-ignore[3]
    autograd_callable: Callable[[Any], Any],  # pyre-ignore[2]
    argnums: Union[int, Tuple[int, ...]] = 0,
    outputnums: Union[int, Tuple[int, ...]] = 0,
) -> Callable[[Any], Any]:
    """Wraps an autograd-implemented function for use with jax.

    The wrapped function returns outputs identical to `autograd_callable`, except
    that outputs in `outputnums` are of type `jnp.ndarray`.

    The arguments to the wrapped function included in `argnums` should be `jnp.ndarray`.
    These can be differentiated with respect to. Other arguments should be as they
    would be provided to `autograd_callable`, i.e. they may be `bool`, etc., and
    generally should not be `jnp.ndarray`. These outputs will have zero gradients.

    Note that the wrapped function is not a proper jax primitive, and does not support
    transformations such as `jax.vmap`.

    Args:
        autograd_callable: The function to be wrapped, which returns either a single
            array or a tuple of arrays.
        argnums: Specifies which inputs can be differentiated with respect to.
        outputnums: Indicates which outputs are to be differentiable.

    Returns:
        The wrapped function for use in a jax setting.
    """
    _argnums: Tuple[int, ...] = _ensure_tuple(argnums)
    _outputnums: Tuple[int, ...] = _ensure_tuple(outputnums)

    @jax.custom_vjp
    def f(*args: Any) -> Any:  # pyre-ignore[3]
        args = tuple([_to_numpy(x) if i in _argnums else x for i, x in enumerate(args)])
        outputs = autograd_callable(*args)
        if isinstance(outputs, tuple):
            return tuple(
                [_to_jax(x) if i in _outputnums else x for i, x in enumerate(outputs)]
            )
        else:
            return _to_jax(outputs)

    def f_fwd(*args: Any) -> Any:  # pyre-ignore[3]
        assert len(list(set(_argnums))) == len(_argnums)
        assert all([i in list(range(len(args))) for i in _argnums])

        # Split out the arguments to be differentiated with respect to, and convert
        # to numpy arrays for use with autograd.
        diff_args, nondiff_args = _split(args, _argnums)
        diff_args = _to_numpy(diff_args)

        # Autograd has some quirks, which preclude its use with functions that return
        # multiple outputs. To work around this, we make a `flat_callable` which
        # flattens all differentiable outputs and returns them in a single array.
        # Nondifferentiable outputs and the output shapes are updated with nonlocal
        # variables.
        #
        # Nonlocal variables updated in `flat_callable`.
        returns_tuple = None
        output_shapes_tree = None
        nondiff_outputs = None

        def flat_callable(*diff_args: Any) -> Any:  # pyre-ignore[3,53]
            # Function which accepts only the arguments to be differentiated with
            # respect to, and returns a single array consisting of the flattened
            # and concatenated differentiable outputs.
            nonlocal returns_tuple
            nonlocal output_shapes_tree
            nonlocal nondiff_outputs

            args = _merge(diff_args, nondiff_args, _argnums)
            outputs = autograd_callable(*args)
            returns_tuple = isinstance(outputs, tuple)
            outputs = _ensure_tuple(outputs)

            assert len(list(set(_outputnums))) == len(_outputnums)
            assert all([i in list(range(len(outputs))) for i in _outputnums])

            diff_outputs, nondiff_outputs = _split(outputs, _outputnums)
            diff_output_flat, output_shapes_tree = _array_from_tree(diff_outputs)
            return diff_output_flat

        autograd_vjp_fn, diff_output_flat = autograd.make_vjp(
            flat_callable,
            argnum=list(range(len(diff_args))),
        )(*diff_args)
        nondiff_grads = [None for _ in nondiff_args]

        diff_outputs = _tree_from_array(diff_output_flat, output_shapes_tree)
        diff_outputs = _to_jax(diff_outputs)

        # Convert any `ArrayBox` objects to numpy arrays.
        nondiff_outputs = pytree.tree_map(_to_array_if_arraybox, nondiff_outputs)

        outputs = _merge(diff_outputs, nondiff_outputs, _outputnums)
        outputs = outputs if returns_tuple else outputs[0]
        return outputs, (pytree.Partial(autograd_vjp_fn), nondiff_grads)

    def f_bwd(res: Any, grads: Any) -> Any:  # pyre-ignore[2,3]
        grads = _ensure_tuple(grads)
        grads, _ = _split(grads, _outputnums)
        grads = _to_numpy(grads)
        grads_array, _ = _array_from_tree(grads)
        autograd_vjp_fn, nondiff_grads = res
        diff_grads = autograd_vjp_fn(grads_array)
        diff_grads = _to_jax(diff_grads)
        return _merge(diff_grads, nondiff_grads, _argnums)

    f.defvjp(f_fwd, f_bwd)
    return f


def _ensure_tuple(x: Union[Value, Tuple[Value, ...]]) -> Tuple[Value, ...]:
    """Ensures that `x` is a tuple, placing it in a tuple if not."""
    return x if isinstance(x, tuple) else (x,)


def _split(
    sequence: Tuple[Value, ...],
    selected_ind: Tuple[int, ...],
) -> Tuple[Tuple[Value, ...], Tuple[Value, ...]]:
    """Splits `sequence` into those with indices in `selected_ind` and those not."""
    selected = tuple([sequence[i] for i in selected_ind])
    other = tuple([sequence[i] for i in range(len(sequence)) if i not in selected_ind])
    return selected, other


def _merge(
    selected: Tuple[Value, ...],
    other: Tuple[Value, ...],
    selected_ind: Tuple[int, ...],
) -> Tuple[Value, ...]:
    """Merges `select` and `other`, undoing a `_split` operation."""
    num = len(selected) + len(other)
    iter_selected = iter(selected)
    iter_other = iter(other)
    return tuple(
        [
            next(iter_selected) if i in selected_ind else next(iter_other)
            for i in range(num)
        ]
    )


@dataclasses.dataclass
class _ShapeAndSize:
    """Stores array shape and size."""

    shape: Tuple[int, ...]
    size: int


def _array_from_tree(tree: PyTree) -> Tuple[onp.ndarray, PyTree]:
    """Flattens a pytree of numpy arrays into a single 1D array.

    Args:
        tree: The pytree of arrays.

    Returns:
        The flattend tree, and a pytree whose leaves are `_ShapeAndSize`
        instances giving the original array leaf shapes and sizes.
    """
    shapes_tree = pytree.tree_map(lambda x: _ShapeAndSize(x.shape, x.size), tree)
    tree_leaves = pytree.tree_leaves(tree)
    flattened = npa.concatenate([x.flatten() for x in tree_leaves])
    return flattened, shapes_tree


def _tree_from_array(array: onp.ndarray, shapes_tree: PyTree) -> PyTree:
    """Restores a pytree from its flattened representation.

    This function undoes a `_array_from_tree` operation, so that a
    pytree `x` and the pytree `_tree_from_array(*_array_from_tree(x))`
    are equivalent.

    Args:
        array: The flat array.
        shapes_tree: The pytree of `_ShapeAndSize` giving the original
            array leaf shapes and sizes.

    Returns:
       The restored pytree.
    """
    shapes_flat, treedef = pytree.tree_flatten(shapes_tree)
    splits = onp.cumsum([s.size for s in shapes_flat])
    split_array = onp.split(array, splits)
    flat_values = [arr.reshape(s.shape) for arr, s in zip(split_array, shapes_flat)]
    return pytree.tree_unflatten(treedef, flat_values)


def _to_jax(tree: PyTree) -> PyTree:
    """Converts numpy arrays in `tree` to jax arrays."""
    return pytree.tree_map(_as_jax_array, tree)


def _to_numpy(tree: PyTree) -> PyTree:
    """Converts jax arrays in `tree` to numpy arrays."""
    return pytree.tree_map(onp.asarray, tree)


def _as_jax_array(
    x: Union[onp.ndarray, autograd.numpy.numpy_boxes.ArrayBox],
) -> jnp.ndarray:
    """Converts `array_or_arraybox` to a jax array."""
    return jnp.asarray(_to_array_if_arraybox(x))


def _to_array_if_arraybox(
    x: Union[onp.ndarray, autograd.numpy.numpy_boxes.ArrayBox],
) -> onp.ndarray:
    """Converts `x` to a numpy array, if it is an `ArrayBox`."""
    if isinstance(x, autograd.numpy.numpy_boxes.ArrayBox):
        return onp.asarray(x._value)
    return x