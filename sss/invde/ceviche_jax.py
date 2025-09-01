# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import functools
import gin
import jax
import jax.numpy as jnp
import numpy as onp
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Callable

from sss.invde.utils.jax_utils import box_downsample, kronecker_upsample, sim_params, extract_fixed_solid_void_pixels, ResponseArray, Density, grayscale_initializer
import sss.invde.utils.jax_autograd_wrapper as autograd_wrapper

AuxDict = Dict[str, Any]
DensityInitializer = Callable[[Tuple[int, int]], jnp.ndarray]
ParamsDict = Dict[str, Any]
JaxSimFn = Callable[
    [jnp.ndarray, Sequence[int], Sequence[float], Optional[int]],
    Tuple[jnp.ndarray, onp.ndarray],
]

_DENSITY_LABEL = "density"
_FIELDS_EZ_LABEL = "fields_ez"
_DISTANCE_TO_WINDOW_LABEL = "distance_to_window"

_DUMMY_WAVELENGTHS_NM = (1000.0,)

class CevicheJaxComponent:
    """Base class for ceviche jax components.

    Attributes:
        model: The `ceviche_challenges.model_base.Model` for the component.
    """

    def __init__(
        self,
        design_resolution_nm: int,
        sim_resolution_nm: int,
        wavelengths_nm: Union[onp.ndarray, Sequence[float]],
        model_constructor: Any,  # pyre-ignore[2]
        density_initializer: DensityInitializer,
        _backend: str = 'ceviche',
    ) -> None:
        """Initializes the component.

        Args:
            design_resolution_nm: The size of a design pixel, in nanometers.
            sim_resolution_nm: The size of a simulation pixel, in nanometers. The sim resolution must
                be evenly divisible by `design_resolution_nm`, or be an integer multiple thereof.
            wavelengths_nm: The default wavelengths for simulation.
            model_constructor: The constructor for the ceviche model.
            density_initializer: The function used to initialize density design variables.
        """

        # Extract the fixed pixels (i.e. pixels which should be kept solid or void) for the
        # sepecific resolution configuration. Also validates that resolutions are compatible.
        self._fixed_solid: jnp.ndarray
        self._fixed_void: jnp.ndarray
        self._fixed_solid, self._fixed_void = _extract_fixed_solid_void(
            design_resolution_nm=design_resolution_nm,
            sim_resolution_nm=sim_resolution_nm,
            model_constructor=model_constructor,
        )
        self._design_shape: Tuple[int, int] = tuple(self._fixed_solid.shape)
        self.model = model_constructor(
            params=sim_params(sim_resolution_nm, wavelengths_nm)
        )
        self._density_initializer = density_initializer
        self._wavelength=wavelengths_nm

        # Construct the jax.grad compatible simulation function for the model.
        self._backend = _backend
        if _backend == 'DDM':
            self.model.init_DDM_workers()
        self._jax_sim_fn: JaxSimFn = self.construct_jax_sim_fn(design_resolution_nm, sim_resolution_nm, self.model)

    def construct_jax_sim_fn(self,
            design_resolution_nm: int,
            sim_resolution_nm: int,
            model: Callable,
        ) -> JaxSimFn:
        """Constructs the jax-compatible simulation function for the model."""
        raise NotImplementedError("Should be implemented in subclasses")


    def response(
        self,
        params: ParamsDict,
        excite_port_idxs: Sequence[int] = (0,),
        wavelengths_nm: Optional[Sequence[float]] = None,
        max_parallelizm: Optional[int] = None,
    ) -> Tuple[ResponseArray, AuxDict]:
        """Computes the response of the component to excitation.

        Args:
            params: The design parameters, as returned by the `init` method.
            excite_port_idxs: Ports to be excited. See `model.simulate`.
            wavelengths_nm: Optional, wavelengths to override the default wavelengths.
            max_parallelizm: Optional, specifies the number of parallel simulations to
                carry out. If `None`, all simulations are done in parallel.

        Returns:
            A `ResponseArray` containing scattering matrix elements, and auxiliary output
            containing the the z-oriented electric field profiles.
        """
        raise NotImplementedError("Should be implemented in subclasses")


    def init(self, key: jax.Array) -> Dict[str, Density]:
        """Returns the initial design, and masks indicating fixed pixels.

        The initial design is random, and value depends on the numpy random
        number generator state.

        args:
            key: The key used in random initialization.

        Returns:
            The initial parameters.
        """
        return {
            _DENSITY_LABEL: Density(
                density=self._density_initializer(  # pyre-ignore[28]
                    key=key,
                    shape=self._design_shape,
                    fixed_solid=self._fixed_solid,
                    fixed_void=self._fixed_void,
                ),
                fixed_solid=jnp.asarray(self._fixed_solid),
                fixed_void=jnp.asarray(self._fixed_void),
            )
        }


def _extract_fixed_solid_void(
    design_resolution_nm: int,
    sim_resolution_nm: int,
    model_constructor: Any,  # pyre-ignore[2]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extracts the fixed solid and void pixels, as appropriate for the resolutions.

    This function also validates that the selected resolution is compatible with the model.

    Args:
        design_resolution_nm: The size of a design pixel, in nanometers.
        sim_resolution_nm: The size of a simulation pixel, in nanometers. The sim resolution must
            be evenly divisible by `design_resolution_nm`, or be an integer multiple thereof.
        model_constructor: The constructor for the ceviche model.

    Returns:
        The fixed solid and void pixels.
    """
    if (
        design_resolution_nm <= sim_resolution_nm
        and sim_resolution_nm % design_resolution_nm != 0
    ):
        raise ValueError(
            f"If `design_resolution_nm` is finer than `sim_resolution_nm`, it must evenly divide "
            f"`sim_resolution_nm`, but got {design_resolution_nm} and {sim_resolution_nm}."
        )
    if (
        design_resolution_nm > sim_resolution_nm
        and design_resolution_nm % sim_resolution_nm != 0
    ):
        raise ValueError(
            f"If `design_resolution_nm` is coarser than `sim_resolution_nm`, it must be an integer "
            f"multiple of `sim_resolution_nm`, but got {design_resolution_nm} and {sim_resolution_nm}."
        )

    if design_resolution_nm <= sim_resolution_nm:
        # If design resolution is equal or finer than the sim resolution, we use
        # a high-resolution model and extract fixed pixels.
        fixed_pixel_resolution_nm = design_resolution_nm
    else:
        # Otherwise, we use a model at the sim resolution, compute the fixed pixels
        # for this model and then downsample.
        fixed_pixel_resolution_nm = sim_resolution_nm

    params = sim_params(fixed_pixel_resolution_nm, _DUMMY_WAVELENGTHS_NM)
    model = model_constructor(params=params)
    density_with_gray_design = model.density(
        onp.ones(model.design_variable_shape) * 0.5
    )
    _fixed_solid, _fixed_void = extract_fixed_solid_void_pixels(
        density_with_gray_design
    )
    fixed_solid = jnp.asarray(_fixed_solid)
    fixed_void = jnp.asarray(_fixed_void)

    if sim_resolution_nm < design_resolution_nm:
        # Ensure the coarse design resolution is compatible with the design shape.
        ratio = design_resolution_nm // sim_resolution_nm
        if any([d % ratio != 0 for d in model.design_variable_shape]):
            raise ValueError(
                f"If `design_resolution_nm` is coarser than `sim_resolution_nm`, the ratio of the "
                f"two must evenly divide the dimensions of the design on the simulation grid, but "
                f"got a ratio {ratio} when the design shape on the simulation grid is "
                f"{model.design_variable_shape}."
            )
        # Apply box downsampling and thresholding to obtain the fixed pixels at
        # the coarser design resolution.
        fixed_solid = box_downsample(fixed_solid.astype(float), ratio) >= 0.5
        fixed_void = box_downsample(fixed_void.astype(float), ratio) >= 0.5

    return fixed_solid, fixed_void


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


class CevicheJaxChallenge:
    """Base class for design problems.

    Ceviche challenges consist of a `CevicheJaxComponent` and windows for the
    transmission into all output ports, given excitation from the 0-th port.

    Two functions, `loss` and `metrics`, are provided to facilitate gradient-
    based optimization of the component, and independent means of evaluating
    the performance of a solution.

    Attributes:
        component: The `CevicheJaxComponent` to be designed.
        min_transmission: The minimum of the transmission window.
        max_transmission: The maximum of the transmission window.
    """

    def __init__(
        self,
        component: CevicheJaxComponent,
        min_transmission: onp.ndarray,
        max_transmission: onp.ndarray,
    ) -> None:
        """Initializes the challenge.

        Args:
            component: The `CevicheJaxComponent` to be designed.
            min_transmission: The minimum transmission target.
            max_transmission: The maximum transmission target.
        """
        self.component = component
        self._min_transmission = min_transmission
        self._max_transmission = max_transmission

    def loss(self, response: ResponseArray) -> float:
        raise NotImplementedError("Should be implemented in subclasses")
        


    # def metrics(
    #     self,
    #     response: ResponseArray,
    #     aux: Optional[AuxDict] = None,
    # ) -> AuxDict:
    #     del aux
    #     return {
    #         _DISTANCE_TO_WINDOW_LABEL: jax_loss.distance_to_window(
    #             transmission=jnp.abs(response.array) ** 2,
    #             min_transmission=self._min_transmission,
    #             max_transmission=self._max_transmission,
    #         )
    #     }


