# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

from typing import Callable, Tuple, Any, Union
import jax
import jax.numpy as jnp
import torch
import jax.tree_util as pytree
import numpy as onp
PyTree = Any
Value = Any

def jax_wrap_torch(
    torch_forward: Callable[[Any], torch.Tensor],
    torch_adjoint: Callable[[Any, torch.Tensor], torch.Tensor],
    argnums: Tuple[int, ...] = (0,),
    outputnums: Union[int, Tuple[int, ...]] = 0,
) -> Callable[[Any], Any]:
    _argnums = _ensure_tuple(argnums)

    @jax.custom_vjp
    def f(*args):
        # Forward pass: convert args to torch, run torch_forward
        args = tuple([_to_torch(x) if i in _argnums else x for i, x in enumerate(args)])
        outputs = torch_forward(*args)
        torch_args = list(args)
        if isinstance(outputs, tuple):
            out = tuple(
                [_to_jax(x) for x in outputs]
            )
        else:
            out = _to_jax(outputs)
        return out

    def f_fwd(*args):
        saved_args = args
        args = tuple([_to_torch(x) if i in _argnums else x for i, x in enumerate(args)])
        outputs = torch_forward(*args)
        torch_args = list(args)
        if isinstance(outputs, tuple):
            out = tuple(
                [_to_jax(x) for x in outputs]
            )
        else:
            out = _to_jax(outputs)

        # print("out in f_fwd: ", out, "len: ", len(out))
        # print("saved_args: ", saved_args, "len: ", len(saved_args))
        return out, (saved_args, out)

    def f_bwd(residuals, grad_output):
        saved_args, forward_output = residuals
        saved_args = tuple([_to_torch(x) if i in _argnums else x for i, x in enumerate(saved_args)])
        grad_output_torch = _to_torch(grad_output)
        forward_output_torch = _to_torch(forward_output)

        torch_grads = torch_adjoint(*saved_args, forward_output_torch, grad_output_torch)
        # torch_grads is a tuple of gradients, with the same structure as the input args to f_fwd
        # Wrap gradients back to JAX, and return in correct arg order
        grads = []
        for i, grad in enumerate(torch_grads):
            if i in _argnums:
                grads.append(jnp.asarray(grad.detach().cpu().numpy()))
            else:
                grads.append(grad)

        return tuple(grads)

    f.defvjp(f_fwd, f_bwd)
    return f


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

def _ensure_tuple(x: Union[Value, Tuple[Value, ...]]) -> Tuple[Value, ...]:
    """Ensures that `x` is a tuple, placing it in a tuple if not."""
    return x if isinstance(x, tuple) else (x,)


def _to_torch(tree: PyTree) -> PyTree:
    """Converts jax arrays in `tree` to numpy arrays."""
    return pytree.tree_map(lambda x: torch.tensor(onp.array(x), requires_grad=False), tree)

def _to_jax(tree: PyTree) -> PyTree:
    """Converts torch tensors in `tree` to jax arrays."""
    def convert_to_jax(x):
        if isinstance(x, torch.Tensor):
            return jnp.asarray(x.cpu())
        else:
            return jnp.asarray(x)
    return pytree.tree_map(convert_to_jax, tree)