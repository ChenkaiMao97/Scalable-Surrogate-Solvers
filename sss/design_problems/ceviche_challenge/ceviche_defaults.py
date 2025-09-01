"""Defines defaults for the Ceviche challenges.

The values essentially replicate those used in "Inverse design of photonic
devices with strict foundry fabrication constraints" by M. F. Schubert et al.
https://doi.org/10.1021/acsphotonics.2c00313

There is a small difference in the beamsplitter, which is slightly wider here.
"""

from typing import Tuple

import ceviche_challenges as cc
import ceviche_challenges.units as u
import numpy as onp

# Parameters for "standard" ceviche challenges, which match the paper.
RESOLUTION_NM: int = 10
WAVELENGTHS_NM: Tuple[float, ...] = (1550,)

# Parameters for "lightweight" ceviche challenges, which are intended
# to run more quickly and are useful for experimentation and testing.
LIGHTWEIGHT_RESOLUTION_NM: int = 40
LIGHTWEIGHT_WAVELENGTHS_NM: Tuple[float, ...] = (1270.0, 1290.0)


_CLADDING_PERMITTIVITY: float = 2.25
_SLAB_PERMITTIVITY: float = 12.25
_WG_WIDTH_NM: int = 400
_WG_LENGTH_NM: int = 800
_PADDING_NM: int = 400
_WG_MODE_PADDING_NM: int = 400
_PORT_PML_OFFSET_NM: int = 200
_INPUT_MONITOR_OFFSET_NM: int = 40
_PML_WIDTH: int = 10

_WG_MIN_WIDTH_NM: int=200
_WG_MAX_WIDTH_NM: int=600
_WG_MIN_SEPARATION_NM: int = 400
_WG_FIELD_SPACING_NM: int = 200

WDM_SPEC = cc.wdm.spec.WdmSpec(
    extent_ij=(4000 * u.nm, 4000 * u.nm),
    input_wg_j=2000 * u.nm,
    output_wgs_j=(1500 * u.nm, 2500 * u.nm),
    wg_width=_WG_WIDTH_NM * u.nm,
    wg_mode_padding=_WG_MODE_PADDING_NM * u.nm,
    # Mode locations are suitable for resolution as coarse as 40 nm.
    input_mode_i=(40 * _PML_WIDTH + _PORT_PML_OFFSET_NM) * u.nm,
    output_mode_i=(4000 - 40 * _PML_WIDTH - _PORT_PML_OFFSET_NM) * u.nm,
    variable_region=((1000 * u.nm, 1000 * u.nm), (3000 * u.nm, 3000 * u.nm)),
    cladding_permittivity=_CLADDING_PERMITTIVITY,
    slab_permittivity=_SLAB_PERMITTIVITY,
    input_monitor_offset=_INPUT_MONITOR_OFFSET_NM * u.nm,
    pml_width=_PML_WIDTH,
)

LIGHTWEIGHT_WDM_SPEC = cc.wdm.spec.WdmSpec(
    extent_ij=(5600 * u.nm, 5600 * u.nm),
    input_wg_j=2800 * u.nm,
    output_wgs_j=(1800 * u.nm, 3800 * u.nm),
    wg_width=_WG_WIDTH_NM * u.nm,
    wg_mode_padding=_WG_MODE_PADDING_NM * u.nm,
    # Mode locations are suitable for resolution as coarse as 40 nm.
    input_mode_i=(40 * _PML_WIDTH + _PORT_PML_OFFSET_NM) * u.nm,
    output_mode_i=(5600 - 40 * _PML_WIDTH - _PORT_PML_OFFSET_NM) * u.nm,
    variable_region=((1200 * u.nm, 1200 * u.nm), (4400 * u.nm, 4400 * u.nm)),
    cladding_permittivity=_CLADDING_PERMITTIVITY,
    slab_permittivity=_SLAB_PERMITTIVITY,
    input_monitor_offset=_INPUT_MONITOR_OFFSET_NM * u.nm,
    pml_width=_PML_WIDTH,
)

BEAM_SPLITTER_SPEC = cc.beam_splitter.spec.BeamSplitterSpec(
    wg_width=_WG_WIDTH_NM * u.nm,
    wg_length=_WG_LENGTH_NM * u.nm,
    wg_separation=1200 * u.nm,  # 1120 nm in [1].
    wg_mode_padding=_WG_MODE_PADDING_NM * u.nm,
    port_pml_offset=_PORT_PML_OFFSET_NM * u.nm,
    variable_region_size=(3200 * u.nm, 2400 * u.nm),  # 3200x2400 nm in [1].
    cladding_permittivity=_CLADDING_PERMITTIVITY,
    slab_permittivity=_SLAB_PERMITTIVITY,
    input_monitor_offset=_INPUT_MONITOR_OFFSET_NM * u.nm,
    design_symmetry=None,
    pml_width=_PML_WIDTH,
)

WAVEGUIDE_BEND_SPEC = cc.waveguide_bend.spec.WaveguideBendSpec(
    wg_width=_WG_WIDTH_NM * u.nm,
    wg_length=_WG_LENGTH_NM * u.nm,
    wg_mode_padding=_WG_MODE_PADDING_NM * u.nm,
    padding=_PADDING_NM * u.nm,
    port_pml_offset=_PORT_PML_OFFSET_NM * u.nm,
    variable_region_size=(2000 * u.nm, 2000 * u.nm),
    cladding_permittivity=_CLADDING_PERMITTIVITY,
    slab_permittivity=_SLAB_PERMITTIVITY,
    input_monitor_offset=_INPUT_MONITOR_OFFSET_NM * u.nm,
    pml_width=_PML_WIDTH,
)

MODE_CONVERTER_SPEC = cc.mode_converter.spec.ModeConverterSpec(
    left_wg_width=_WG_WIDTH_NM * u.nm,
    left_wg_mode_padding=_WG_MODE_PADDING_NM * u.nm,
    left_wg_mode_order=1,
    right_wg_width=_WG_WIDTH_NM * u.nm,
    right_wg_mode_padding=_WG_MODE_PADDING_NM * u.nm,
    right_wg_mode_order=2,
    wg_length=_WG_LENGTH_NM * u.nm,
    padding=_PADDING_NM * u.nm,
    port_pml_offset=_PORT_PML_OFFSET_NM * u.nm,
    variable_region_size=(2000 * u.nm, 2000 * u.nm),
    cladding_permittivity=_CLADDING_PERMITTIVITY,
    slab_permittivity=_SLAB_PERMITTIVITY,
    input_monitor_offset=_INPUT_MONITOR_OFFSET_NM * u.nm,
    pml_width=_PML_WIDTH,
)

POWER_SPLITTER_SPEC = cc.wdm.spec.WdmSpec(
    extent_ij=(4000 * u.nm, 4000 * u.nm),
    input_wg_j=2000 * u.nm,
    output_wgs_j=(1600 * u.nm, 2400 * u.nm),
    wg_width=_WG_WIDTH_NM * u.nm,
    wg_mode_padding=_WG_MODE_PADDING_NM * u.nm,
    # Mode locations are suitable for resolution as coarse as 40 nm.
    input_mode_i=(40 * _PML_WIDTH + _PORT_PML_OFFSET_NM) * u.nm,
    output_mode_i=(4000 - 40 * _PML_WIDTH - _PORT_PML_OFFSET_NM) * u.nm,
    variable_region=((1200 * u.nm, 1200 * u.nm), (2800 * u.nm, 2800 * u.nm)),
    cladding_permittivity=_CLADDING_PERMITTIVITY,
    slab_permittivity=_SLAB_PERMITTIVITY,
    input_monitor_offset=_INPUT_MONITOR_OFFSET_NM * u.nm,
    pml_width=_PML_WIDTH,
)

# ----------------------------------------------------------------
# Target transmission values, from table 1 of the above reference.
# ----------------------------------------------------------------


def _from_dB(dB_value: float) -> float:
    return 10.0 ** (dB_value / 10.0)


WDM_MIN_TRANSMISSION: onp.ndarray = onp.array(
    [
        # S11  S12             S13
        [[0.0, _from_dB(-3.0), 0.0]],  # First wavelength band
        [[0.0, 0.0, _from_dB(-3.0)]],  # Second wavelength band
    ]
)
WDM_MAX_TRANSMISSION: onp.ndarray = onp.array(
    [
        [[_from_dB(-20.0), 1.0, _from_dB(-20.0)]],
        [[_from_dB(-20.0), _from_dB(-20.0), 1.0]],
    ]
)

BEAM_SPLITTER_MIN_TRANSMISSION: onp.ndarray = onp.array(
    [[[0.0, _from_dB(-3.5), _from_dB(-3.5), 0.0]]]
)
# The maximum tranmsission is chosen so that when summed with the minimum
# transmission target it yields unity.
BEAM_SPLITTER_MAX_TRANSMISSION: onp.ndarray = onp.array(
    [[[_from_dB(-20.0), 1.0 - _from_dB(-3.5), 1.0 - _from_dB(-3.5), _from_dB(-20.0)]]]
)

WAVEGUIDE_BEND_MIN_TRANSMISSION: onp.ndarray = onp.array([[[0.0, _from_dB(-0.5)]]])
WAVEGUIDE_BEND_MAX_TRANSMISSION: onp.ndarray = onp.array([[[_from_dB(-20.0), 1.0]]])

MODE_CONVERTER_MIN_TRANSMISSION: onp.ndarray = onp.array([[[0.0, _from_dB(-0.5)]]])
MODE_CONVERTER_MAX_TRANSMISSION: onp.ndarray = onp.array([[[_from_dB(-20.0), 1.0]]])

POWER_SPLITTER_MIN_TRANSMISSION: onp.ndarray = onp.array(
    [[[0.0, _from_dB(-3.5), _from_dB(-3.5)]]]
)
POWER_SPLITTER_MAX_TRANSMISSION: onp.ndarray = onp.array(
    [[[_from_dB(-20.0), 1.0 - _from_dB(-3.5), 1.0 - _from_dB(-3.5)]]]
)

GENERAL_BLOCK_MIN_TRANSMISSION: onp.ndarray = onp.array([[[0.0, _from_dB(-0.5)]]])
GENERAL_BLOCK_MAX_TRANSMISSION: onp.ndarray = onp.array([[[_from_dB(-20.0), 1.0]]])