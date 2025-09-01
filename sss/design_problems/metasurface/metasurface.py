# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import functools
import jax.tree_util
import jax.numpy as jnp
import numpy as onp
from typing import Sequence, Union, Callable, Optional, Tuple, Any, Dict
from dataclasses import dataclass

from sss.invde.ceviche_jax import CevicheJaxComponent, CevicheJaxChallenge, DensityInitializer
from sss.invde.utils.jax_utils import _get_default_initializer
from sss.design_problems.metasurface.metasurface_model import MetasurfaceModel

import sss.invde.utils.jax_autograd_wrapper as autograd_wrapper
import sss.invde.utils.jax_torch_wrapper as torch_wrapper

import gin

_DENSITY_LABEL = "density"
_FIELDS_EZ_LABEL = "fields_ez"
_FOM_LABEL = "FOM"

@dataclass
class ResponseArray:
    array: jnp.ndarray

jax.tree_util.register_pytree_node(
    nodetype=ResponseArray,
    flatten_func=lambda s: ( (s.array,), (None,)),
    unflatten_func=lambda aux, array: ResponseArray(*array),
)

@gin.configurable
class MetasurfaceComponent(CevicheJaxComponent):
    """metasurface component with arbitrary number of ports and locations"""
    def __init__(
        self,
        design_resolution_nm: int,
        sim_resolution_nm: int,
        wavelengths_nm: Union[onp.ndarray, Sequence[float]],
        focus_positions_nm: Sequence[Tuple[float, float]],
        density_initializer_getter: Callable = _get_default_initializer,
        backend: str = 'ceviche',
    ) -> None:
        super().__init__(
            design_resolution_nm=design_resolution_nm,
            sim_resolution_nm=sim_resolution_nm,
            wavelengths_nm=wavelengths_nm,
            model_constructor=functools.partial(MetasurfaceModel, _backend=backend, focus_positions_nm=focus_positions_nm, wavelengths_nm=wavelengths_nm),
            density_initializer=density_initializer_getter(),
            _backend=backend
        )
    
    def construct_jax_sim_fn(self,
            design_resolution_nm: int,
            sim_resolution_nm: int,
            model: Callable,
        ):
        """Constructs the jax-compatible simulation function for the model."""
        # Wrap the model simulation function for use with jax. It has signature:
        # Here, the design must be at the simulation grid resolution.
        if self._backend == 'ceviche':
            _jax_wrapped_sim_fn = autograd_wrapper.jax_wrap_autograd(
                model.simulate, argnums=0, outputnums=0
            )
        elif self._backend == 'DDM':
            _jax_wrapped_sim_fn = torch_wrapper.jax_wrap_torch(
                model.simulate, model.simulate_adjoint, argnums=0
            )
        return _jax_wrapped_sim_fn
    
    def response(
        self,
        params,
        wavelengths_nm: Optional[Sequence[float]] = None,
    ):
        density = params[_DENSITY_LABEL].density

        if wavelengths_nm is None:
            wavelengths_nm = tuple(self._wavelength)
        opt_coupling_eff, FOM_coupling_eff, ez = self._jax_sim_fn(
            density, wavelengths_nm
        )
        opt_coupling_eff = ResponseArray(array = opt_coupling_eff)
        return opt_coupling_eff, {_FIELDS_EZ_LABEL: ez, _FOM_LABEL: FOM_coupling_eff}



@gin.configurable
class MetasurfaceChallenge(CevicheJaxChallenge):
    """General Block design challenge."""
    def __init__(
        self,
        design_resolution_nm,
        sim_resolution_nm,
        wavelengths_nm: Sequence[float],
        density_initializer_getter: Callable = _get_default_initializer,
        focus_positions_nm: Sequence[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(
            component=MetasurfaceComponent(
                design_resolution_nm=design_resolution_nm,
                sim_resolution_nm=sim_resolution_nm,
                wavelengths_nm=wavelengths_nm,
                density_initializer_getter=density_initializer_getter,
                focus_positions_nm=focus_positions_nm
            ),
            min_transmission=jnp.asarray([0.0] * len(wavelengths_nm)),
            max_transmission=jnp.asarray([1.0] * len(wavelengths_nm)),
        )
        self._focus_positions_nm = focus_positions_nm
    
    def loss(self, response) -> float:
        return jnp.sum((jnp.abs(response.array) ** 2 - self._max_transmission)**2)  
