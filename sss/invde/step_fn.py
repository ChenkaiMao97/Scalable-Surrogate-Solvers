# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

"""
File contains step functions for pol sort ccsa optimizer. 
"""
import os, gin
import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as onp
from sss.invde.utils.utils import CevicheState
from sss.invde.ceviche_jax import CevicheJaxChallenge

import matplotlib.pyplot as plt

# Set default font sizes for matplotlib
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12
})

def mse_loss_fn(
    latents: Dict[str, Any],
    challenge: CevicheJaxChallenge,
    latent_to_param_fn: Callable,
    state: CevicheState
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Any, Dict[str, Any]]]:
    params = latent_to_param_fn(latents, beta=state.beta)
    response, aux = challenge.component.response(params)
    loss = challenge.loss(response)
    return loss, (response, aux, params)

@gin.configurable
def ceviche_ccsa_step_fn(
    state: CevicheState,
    decode_fn: Callable,
    latent_to_param_fn: Callable,
    latent_shape: Tuple[int, int],
    challenge: CevicheJaxChallenge,
    loss_fn: Callable=mse_loss_fn,
    log_fn: Callable=None,
) -> Tuple[Callable, Callable, CevicheState]:
    """Build an nlopt-compatible cost function using an MSE formulation.
    Returns:
        The nlopt-friendly cost function and the loss function.
    """
    def _nlopt_cost_function(x: onp.ndarray, grad: onp.ndarray):
        # update latent in state
        state.latents["density"].density = jnp.reshape(jnp.asarray(x), latent_shape)
        
        (loss_value, (response, aux, params)), my_grad = jax.value_and_grad(loss_fn, has_aux=True)(state.latents, challenge, latent_to_param_fn, state)

        if grad.size > 0:
            grad[:] = my_grad["density"].density.flatten()

        value = onp.mean(loss_value)
        print(f"step: {state.step}, loss: {value}")

        # update state params, loss and step
        state.params = params
        state.loss.append(float(value))
        state.step += 1

        # log step
        log_fn(my_grad,params,response,aux)

        return float(value)

    return _nlopt_cost_function, loss_fn