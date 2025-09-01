import gin
from ceviche_challenges import units as u
import h5py

from typing import List, Tuple, Dict, Any, Union, Callable, Sequence, Optional
import jax
import jax.numpy as jnp
import numpy as onp

from dataclasses import dataclass, field
import ceviche_challenges as cc

from sss.design_problems.ceviche_challenge.ceviche_general_model import GeneralBlockSpec
from sss.design_problems.ceviche_challenge.general_block import GeneralBlockChallenge
from sss.design_problems.grating_coupler.grating_coupler import GratingCouplerChallenge
from sss.design_problems.metasurface.metasurface import MetasurfaceChallenge


from numpy.random import choice
import math

def _from_dB(dB_value: float) -> float:
    return 10.0 ** (dB_value / 10.0)

@dataclass
class CevicheState:
    step: int
    beta: int
    beta_schedule_step: int
    latents: onp.ndarray
    params: onp.ndarray
    latent_shape: Any
    density_shape: Any
    optimizer_state: Any = None
    loss: List[float] = field(default_factory=list)

@gin.configurable
def get_ceviche_general_block_challenge(
    key: jax.Array,
    wavelengths_nm: List[float],
    left_wg_options: List[int] = [1,2], # input port is always placed on the left
    top_wg_options: List[int] = [0,1,2],
    right_wg_options: List[int] = [0,1,2],
    bottom_wg_options: List[int] = [0,1,2],
    num_wg_weights: List[int] = [1,1,1], 
    randomize_width: bool = False,
    randomize_center: bool = True,
    symmetric_two_wg: bool = True,
    variable_region_size_options: List[Tuple[float]] = None,
    variable_region_size_nm: Tuple[float] = (2000, 2000),
    wg_length_nm: float = 800,
    wg_model_order_options: List[int] = [0,1],
    wg_mode_weights: List[int] = [1,1], 
    wg_min_width_nm: float = 200,
    wg_max_width_nm: float = 600,
    wg_min_separation_nm: float = 800,
    resolution: float = 10,
    save_wg_io_fields: bool = False,
    constrained_initialization = False,
    random_objective=False,
    max_transmission_ports=None,
    cladding_permittivity=2.25,
    slab_permittivity=12.25,
    port_pml_offset_nm=200,
    input_monitor_offset_nm=40,
    wg_mode_padding_nm=400,
    wg_field_spacing_nm=200,
    wg_to_edge_pixels = None,
):
    """
                      ################
                      ##  top side  ##
                      ################
        ############# ---------------- ##############
        # left side # | design block | # right side #
        ############# ---------------- ##############
                      ################
    y ^               #  bottom side #
      |               ################
      -->
         x
    (x,y corresponds to numpy array row and column, when plotting numpy arrays, rot90 first)
    Arguments:
        <side>_wg_options: possible number of waveguides on each side of the central design block
        num_wg_weights: weights for the possibility of choosing each number of waveguides
        randomize_width: if True, randomize the waveguide widths
        randomize_center: if True, ranodmize the waveguide positions, if False, 1 waveguide will be placed in the center and two waveguides on two sides
        symmetric_two_wg: if True, two waveguides will be placed symmetrically across the center
        variable_region_size_nm: design region sizes: (n,m)
        min_wg_separation_nm: spacing between waveguides in nm
        wg_length_nm: waveguide length in nm
        wg_mode_order: order of the waveguide mode, defaults to be all 1.
        wg_min_width_nm: minimum waveguide width in nm
        wg_max_width_nm: maximum waveguide width in nm
        wg_min_separation_nm: minimum separation between waveguides
        save_wg_io_fields: if True, mask out the input wg field and the output wg fields and store as data (to be used for model training)
    """
    if variable_region_size_options is not None:
        random_index = onp.random.randint(0, len(variable_region_size_options))
        variable_region_size_nm = variable_region_size_options[random_index]

    assert len(left_wg_options)*len(top_wg_options)*len(top_wg_options)*len(top_wg_options)>0, "please provide at least 1 options for number of waveguides on each side"
    def choice_with_p(options, unscaled_weights):
        assert max(options) < len(unscaled_weights)
        weights = onp.array([unscaled_weights[k] for k in options])
        return choice(options, p=weights/onp.sum(weights))
    num_wg_left, num_wg_right, num_wg_top, num_wg_bottom = 0,0,0,0
    while num_wg_left + num_wg_right + num_wg_top + num_wg_bottom == 0:
        num_wg_left, num_wg_right, num_wg_top, num_wg_bottom = choice_with_p(left_wg_options, num_wg_weights), choice_with_p(right_wg_options, num_wg_weights), choice_with_p(top_wg_options, num_wg_weights), choice_with_p(bottom_wg_options, num_wg_weights)

    def num_of_wg_to_specs(key, num, side_length):
        side_pixels = round(side_length/resolution)
        assert side_length/resolution - side_pixels < 1e-3
        if num == 0:
            return ()
        elif num == 1:
            if randomize_width:
                rng, key = jax.random.split(key)
                width_pixel = jax.random.randint(rng, (1,), round(wg_min_width_nm/resolution), round(wg_max_width_nm/resolution)+1)[0]
            else:
                width_pixel = round(1/2*(wg_max_width_nm+wg_min_width_nm)/resolution)
            if randomize_center:
                rng, key = jax.random.split(key)
                start_pixel = jax.random.randint(rng, (1,), 0, side_pixels-width_pixel)[0]
                center_offset_pixels = start_pixel + width_pixel/2 - side_pixels/2 # migth be half integers
            else:
                center_offset_pixels = 0
            return ((center_offset_pixels * resolution * u.nm, width_pixel * resolution * u.nm),)
        elif num == 2:
            if symmetric_two_wg:
                if randomize_width:
                    max_possible_width_pixel = min(round(wg_max_width_nm/resolution), int((side_length - wg_min_separation_nm)/2/resolution))
                    assert max_possible_width_pixel >= round(wg_min_width_nm/resolution), "impossible physical dimensions, check side length, wg width and spacing"
                    rng, key = jax.random.split(key)
                    width_pixel = jax.random.randint(rng, (1,), round(wg_min_width_nm/resolution), max_possible_width_pixel+1)[0]
                else:
                    width_pixel = round(1/2*(wg_max_width_nm+wg_min_width_nm)/resolution)
                
                if randomize_center:
                    bass_offset_pixel = 0.5 if side_pixels%2 else 0
                    side_min_offset_pixel = math.ceil(wg_min_separation_nm/resolution/2 - bass_offset_pixel)
                    side_max_offset_pixel = round(side_pixels/2-bass_offset_pixel-width_pixel)
                    assert side_max_offset_pixel>=side_min_offset_pixel, "impossible physical dimensions, check side length, wg width and spacing"
                    rng, key = jax.random.split(key)
                    side_offset_pixel = jax.random.randint(rng, (1,), side_min_offset_pixel, side_max_offset_pixel+1)[0]
                    center_offset_pixel = side_offset_pixel + bass_offset_pixel + width_pixel/2 # migth be half integers
                else:
                    center_offset_pixel = side_pixels/2-width_pixel/2
            else:
                raise NotImplementedError("asymmetrical waveguide arrangements not implemented yet")
            return ((center_offset_pixel * resolution * u.nm, width_pixel * resolution * u.nm), (-center_offset_pixel * resolution * u.nm, width_pixel * resolution * u.nm))
        else:
            # for more than 2 waveguides on one side, Assume they are evenly spaced for now:
            if randomize_width:
                max_possible_width_pixel = min(round(wg_max_width_nm/resolution), int((side_length - wg_min_separation_nm*(num-1))/num/resolution))
                assert max_possible_width_pixel >= round(wg_min_width_nm/resolution), "impossible physical dimensions, check side length, wg width and spacing"
                rng, key = jax.random.split(key)
                width_pixel = jax.random.randint(rng, (1,), round(wg_min_width_nm/resolution), max_possible_width_pixel+1)[0]
            else:
                width_pixel = round(1/2*(wg_max_width_nm+wg_min_width_nm)/resolution)
            
            nonlocal wg_to_edge_pixels
            if wg_to_edge_pixels is None:
                wg_to_edge_pixels = (side_pixels - num*width_pixel) / (num+1)
            air_pixels = side_pixels - 2*wg_to_edge_pixels - num*width_pixel
            spec = [((wg_to_edge_pixels + (1/2+i) * width_pixel + round(i*air_pixels/(num-1))-side_pixels/2) * resolution * u.nm, width_pixel * resolution * u.nm) for i in range(num)][::-1]
            return tuple(spec)
                    
    # split keys::
    keys = jax.random.split(key, num=5)
    key = keys[4]
    left_wg_specs, top_wg_specs, right_wg_specs, bottom_wg_specs = num_of_wg_to_specs(keys[0], num_wg_left, variable_region_size_nm[1]), \
                                                                   num_of_wg_to_specs(keys[1], num_wg_top, variable_region_size_nm[0]), \
                                                                   num_of_wg_to_specs(keys[2], num_wg_right, variable_region_size_nm[1]), \
                                                                   num_of_wg_to_specs(keys[3], num_wg_bottom, variable_region_size_nm[0])
    total_num_wgs = len(left_wg_specs)+len(right_wg_specs)+len(top_wg_specs)+len(bottom_wg_specs)
    
    # assume the input waveguide always have 1st order
    wg_mode_orders = [1] + [choice_with_p(wg_model_order_options, wg_mode_weights)+1 for _ in range(num_wg_left+num_wg_top+num_wg_right+num_wg_bottom-1)]

    spec = GeneralBlockSpec(
        variable_region_size = (variable_region_size_nm[0] * u.nm, variable_region_size_nm[1] * u.nm),
        wg_length = wg_length_nm * u.nm,
        wg_min_width = wg_min_width_nm * u.nm,
        wg_max_width = wg_max_width_nm * u.nm,
        wg_min_separation = wg_min_separation_nm * u.nm,
        wg_mode_orders = wg_mode_orders,
        left_wg_specs = left_wg_specs,
        right_wg_specs = right_wg_specs,
        top_wg_specs = top_wg_specs,
        bottom_wg_specs = bottom_wg_specs,
        input_port_idx = onp.random.randint(total_num_wgs) if random_objective else 0,
        cladding_permittivity = cladding_permittivity,
        slab_permittivity = slab_permittivity,
        port_pml_offset = port_pml_offset_nm * u.nm,
        input_monitor_offset = input_monitor_offset_nm * u.nm,
        wg_mode_padding = wg_mode_padding_nm * u.nm,
        wg_field_spacing_nm = wg_field_spacing_nm * u.nm
    )

    _min_transmission = []
    _max_transmission = []
    for i in range(len(wavelengths_nm)):
        if random_objective:
            trans_weight = softmax(5*onp.random.random(size=(1,total_num_wgs)))
            _max_transmission.append(trans_weight)
            _min_transmission.append(trans_weight*0.5)
        elif max_transmission_ports is not None:
            assert len(max_transmission_ports) == len(wavelengths_nm) and max(max_transmission_ports) < total_num_wgs
            maximum_output_port = max_transmission_ports[i]
            _min_t = [[0.0]*total_num_wgs]
            _max_t = [[_from_dB(-20.0)]*total_num_wgs]
            _min_t[0][maximum_output_port] = _from_dB(-0.1)
            _max_t[0][maximum_output_port] = 1.0
            _min_transmission.append(_min_t)
            _max_transmission.append(_max_t)
        else:
            # if only one port, make it an mirror device (go back to either the same wg, or the other wg if there are two):
            if len(right_wg_specs) == 0 and len(top_wg_specs) == 0 and len(bottom_wg_specs)==0:
                if len(left_wg_specs) == 1:
                    _min_transmission.append([[_from_dB(-0.5)]])
                    _max_transmission.append([[1.0]])
                else:
                    _min_transmission.append([[0.0, _from_dB(-0.5)]])
                    _max_transmission.append([[_from_dB(-20.0), 1.0]])
            else: # randomly select one output ports on 3 other sides to maximize:
                rng, key = jax.random.split(key)
                maximum_output_port = jax.random.randint(rng, (1,), len(left_wg_specs), total_num_wgs)[0]
                _min_t = [[0.0]*total_num_wgs]
                _max_t = [[_from_dB(-20.0)]*total_num_wgs]
                _min_t[0][maximum_output_port] = _from_dB(-0.1)
                _max_t[0][maximum_output_port] = 1.0
                _min_transmission.append(_min_t)
                _max_transmission.append(_max_t)
            
            
    _min_transmission = onp.array(_min_transmission)
    _max_transmission = onp.array(_max_transmission)

    # optional to have a initializer with length scale constraints:
    if constrained_initialization:
        rng, key = jax.random.split(key)
        design = constrained_random_data_gen_wrapper(key=rng)
        initializer = functools.partial(  # pyre-ignore[5]
            initializers.fixed_initializer,
            pattern=onp.asarray(design['grayscale_img'])
        )

        return GeneralBlockChallenge(
                    wavelengths_nm = wavelengths_nm,
                    spec = spec, 
                    min_transmission=_min_transmission, 
                    max_transmission=_max_transmission, 
                    save_wg_io_fields=save_wg_io_fields,
                    density_initializer=initializer)
    else:
        return GeneralBlockChallenge(
                    wavelengths_nm = wavelengths_nm,
                    spec = spec,
                    min_transmission=_min_transmission,
                    max_transmission=_max_transmission,
                    save_wg_io_fields=save_wg_io_fields)

@gin.configurable
def get_ceviche_grating_coupler_challenge(
    key: jax.Array,
    wavelengths_mm: List[float],
):
    return GratingCouplerChallenge(wavelengths_mm=wavelengths_mm)

@gin.configurable
def get_ceviche_metasurface_challenge(
    key: jax.Array,
    wavelengths_nm: List[float],
    focus_positions_nm: List[Tuple[float, float]],
):
    return MetasurfaceChallenge(wavelengths_nm=wavelengths_nm, focus_positions_nm=focus_positions_nm)
