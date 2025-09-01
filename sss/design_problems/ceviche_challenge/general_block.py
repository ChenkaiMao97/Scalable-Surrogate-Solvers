# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import functools
import jax.numpy as jnp
import numpy as onp
from typing import Sequence, Union, Callable, Optional, Tuple, Any, Dict

from sss.invde.utils.jax_utils import _get_default_initializer, _rasterize_to_sim_grid
from sss.invde.ceviche_jax import CevicheJaxComponent, CevicheJaxChallenge, DensityInitializer
from sss.design_problems.ceviche_challenge.ceviche_general_model import GeneralBlockSpec, GeneralBlockModel
from sss.design_problems.ceviche_challenge import ceviche_defaults

import gin

import sss.invde.utils.jax_autograd_wrapper as autograd_wrapper
import sss.invde.utils.jax_torch_wrapper as torch_wrapper
from sss.invde.utils.jax_utils import ResponseArray
JaxSimFn = Callable[
    [jnp.ndarray, Sequence[int], Sequence[float], Optional[int]],
    Tuple[jnp.ndarray, onp.ndarray],
]
ParamsDict = Dict[str, Any]
AuxDict = Dict[str, Any]
_DENSITY_LABEL = "density"
_FIELDS_EZ_LABEL = "fields_ez"
# _DISTANCE_TO_WINDOW_LABEL = "distance_to_window"
# _DUMMY_WAVELENGTHS_NM = (1000.0,)

@gin.configurable
class GeneralBlockComponent(CevicheJaxComponent):
    """general block component with arbitrary number of ports and locations"""
    def __init__(
        self,
        design_resolution_nm: int,
        sim_resolution_nm: int,
        wavelengths_nm: Union[onp.ndarray, Sequence[float]],
        spec: GeneralBlockSpec,
        density_initializer_getter: Callable = _get_default_initializer,
        backend: str = 'ceviche',
        save_wg_io_fields: bool = False
    ) -> None:
        super().__init__(
            design_resolution_nm=design_resolution_nm,
            sim_resolution_nm=sim_resolution_nm,
            wavelengths_nm=wavelengths_nm,
            model_constructor=functools.partial(GeneralBlockModel, spec=spec, save_wg_io_fields=save_wg_io_fields, _backend=backend),
            density_initializer=density_initializer_getter(),
            _backend=backend
        )
    
    def construct_jax_sim_fn(self,
            design_resolution_nm: int,
            sim_resolution_nm: int,
            model: Callable,
        ) -> JaxSimFn:
        """Constructs the jax-compatible simulation function for the model."""
        # Wrap the model simulation function for use with jax. It has signature:
        # `f(design, excite_port_idxs, wavelengths_nm, max_parallelizm) -> (s_params, fields_ez)`
        # Here, the design must be at the simulation grid resolution.
        if self._backend == 'ceviche':
            _jax_wrapped_sim_fn = autograd_wrapper.jax_wrap_autograd(
                model.simulate, argnums=0, outputnums=0
            )
        elif self._backend == 'DDM':
            _jax_wrapped_sim_fn = torch_wrapper.jax_wrap_torch(
                model.simulate, model.simulate_adjoint, argnums=0
            )

        def _jax_sim_fn(  # pyre-ignore[53]
            design: jnp.ndarray,
            excite_port_idxs: Sequence[int],
            wavelengths_nm: Sequence[float],
            max_parallelizm: Optional[int],
        ) -> Tuple[jnp.ndarray, onp.ndarray]:
            design = _rasterize_to_sim_grid(design, design_resolution_nm, sim_resolution_nm)
            return _jax_wrapped_sim_fn(
                design, excite_port_idxs, wavelengths_nm, max_parallelizm
            )

        return _jax_sim_fn

    def response(
        self,
        params: ParamsDict,
        excite_port_idxs: Sequence[int] = (0,),
        wavelengths_nm: Optional[Sequence[float]] = None,
        max_parallelizm: Optional[int] = None,
    ) -> Tuple[ResponseArray, AuxDict]:
        density = params[_DENSITY_LABEL].density

        if wavelengths_nm is None:
            wavelengths_nm = tuple(self._wavelength)
        if max_parallelizm is None:
            max_parallelizm = len(excite_port_idxs) * len(wavelengths_nm)
        s_params, fields_ez = self._jax_sim_fn(
            density, excite_port_idxs, wavelengths_nm, max_parallelizm
        )
        s_params = ResponseArray(
            array=s_params,
            wavelengths_nm=wavelengths_nm,
            excite_port_idxs=excite_port_idxs,
            output_port_idxs=tuple(range(s_params.shape[-1])),
        )
        return s_params, {_FIELDS_EZ_LABEL: fields_ez}

@gin.configurable
class GeneralBlockChallenge(CevicheJaxChallenge):
    """General Block design challenge."""

    def __init__(
        self,
        design_resolution_nm: int = ceviche_defaults.RESOLUTION_NM,
        sim_resolution_nm: int = ceviche_defaults.RESOLUTION_NM,
        wavelengths_nm: Sequence[float] = ceviche_defaults.WAVELENGTHS_NM,
        spec: GeneralBlockSpec = GeneralBlockSpec(),
        density_initializer_getter: Callable = _get_default_initializer,
        min_transmission: onp.ndarray = ceviche_defaults.GENERAL_BLOCK_MIN_TRANSMISSION,
        max_transmission: onp.ndarray = ceviche_defaults.GENERAL_BLOCK_MAX_TRANSMISSION,
        save_wg_io_fields: bool = False
    ) -> None:
        super().__init__(
            component=GeneralBlockComponent(
                design_resolution_nm=design_resolution_nm,
                sim_resolution_nm=sim_resolution_nm,
                wavelengths_nm=wavelengths_nm,
                spec=spec,
                density_initializer_getter=density_initializer_getter,
                save_wg_io_fields=save_wg_io_fields
            ),
            min_transmission=min_transmission,
            max_transmission=max_transmission,
        )
    
    def loss(self, response: ResponseArray) -> float:
        res = jnp.sum((jnp.abs(response.array) ** 2 - self._max_transmission)**2)
        return res