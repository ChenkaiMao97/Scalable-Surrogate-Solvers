# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

"""Custom general block component"""

import torch
import numpy as np

from ceviche_challenges import units as u
from ceviche_challenges import defs
from ceviche_challenges import model_base
from ceviche_challenges import modes
from ceviche_challenges import params as _params
from ceviche_challenges.scattering import calculate_amplitudes as calculate_amplitudes_cc
from ceviche_challenges.model_base import _wavelengths_nm_to_omegas

import ceviche
from ceviche.constants import C_0, EPSILON_0, MU_0
import concurrent.futures

from sss.design_problems.ceviche_challenge import ceviche_defaults
from sss.utils.GPU_worker_utils import solver_worker
from sss.invde.utils.torch_functions import calculate_amplitudes as calculate_amplitudes_torch
from sss.invde.utils.torch_functions import make_torch_epsilon_r
from sss.utils.PDE_utils import Ez_to_Hx, Ez_to_Hy, maxwell_robin_residue

import itertools
import torch.multiprocessing as mp
import threading
from typing import Tuple, List, Sequence, Optional

import gin

Q = u.Quantity  # pylint: disable=invalid-name
QSequence = Sequence[Q]  # pylint: disable=invalid-name

def get_ridges(vec, ridge_value=1):
  # binarize vec:
  vec = np.where(vec<0.5, 0, 1)
  assert vec[0] == 1-ridge_value and len(vec.shape) == 1 and ridge_value in [0, 1]

  starts, ends = [], []
  last_value = vec[0]
  for i in range(1, vec.shape[0]):
    if vec[i] != last_value:
      last_value = vec[i]
      if vec[i] == ridge_value:
        starts.append(i)
      else:
        ends.append(i)

  return starts, ends

def source_scale(wl, dL):
  return 1j*2*np.pi*C_0*dL**2/wl*EPSILON_0


@gin.configurable
class GeneralBlockSpec:
  """Parameters specifying the physical structure of the mode converter.

  Attributes:
    variable_region_size: a sequence specifying the size of the design variable region.
    pml_width: the integer number of PML cells we should use within the volume at the ends of each axis.
    wg_length: the length of the waveguides entering on four sides of the design region.
    port_pml_offset: the offset between the ports and the PML
    cladding_permittivity: the relative permittivity of the cladding surrounding the slab.  Also used as the permittivity for design pixels valued `0`.
    slab_permittivity: the relative permittivity within the slab, as well as the permittivity value for design pixels valued `1`.
    input_monitor_offset: The offset of the input monitor  from the input source, along i.
    wg_min_width: minimum width for waveguide
    wg_max_width: maximum width for waveguide
    wg_min_separation: the minimum spacing between adjascent waveguides
    wg_mode_padding: padding space added to the waveguide width for computing the modes
    wg_mode_orders: list of orders for each waveguide
    <side>_wg_specs: list of tuples of two lengths: (center_offset, width), first is offset from the block side-center, second is width
  """
  def __init__(self,
              variable_region_size: QSequence = (2000 * u.nm, 2000 * u.nm),
              pml_width: int = ceviche_defaults._PML_WIDTH,
              wg_length: Q = ceviche_defaults._WG_LENGTH_NM * u.nm,
              port_pml_offset: Q = ceviche_defaults._PORT_PML_OFFSET_NM * u.nm,
              cladding_permittivity: float = ceviche_defaults._CLADDING_PERMITTIVITY,
              slab_permittivity: float = ceviche_defaults._SLAB_PERMITTIVITY,
              input_monitor_offset: Q = ceviche_defaults._INPUT_MONITOR_OFFSET_NM * u.nm,
              wg_min_width: Q = ceviche_defaults._WG_MIN_WIDTH_NM * u.nm,
              wg_max_width: Q = ceviche_defaults._WG_MAX_WIDTH_NM * u.nm,
              wg_min_separation: Q = ceviche_defaults._WG_MIN_SEPARATION_NM * u.nm,
              wg_mode_padding: Q = ceviche_defaults._WG_MODE_PADDING_NM * u.nm,
              wg_mode_orders: List[int] = [1,1,1,1],
              # variable waveguides on 4 sides:
              left_wg_specs: Tuple[Tuple[Q,Q]] = ((0 * u.nm, 400 * u.nm),),
              right_wg_specs: Tuple[Tuple[Q,Q]] = ((0 * u.nm, 400 * u.nm),),
              top_wg_specs: Tuple[Tuple[Q,Q]] = ((0 * u.nm, 400 * u.nm),),
              bottom_wg_specs: Tuple[Tuple[Q,Q]] = ((0 * u.nm, 400 * u.nm),),
              # variable for wg_io_field annotation
              wg_field_spacing_nm: Q = ceviche_defaults._WG_FIELD_SPACING_NM * u.nm,
              input_port_idx: int = 0
  ):
    self.variable_region_size = variable_region_size
    self.pml_width = pml_width
    self.wg_length = wg_length
    self.port_pml_offset = port_pml_offset
    self.cladding_permittivity = cladding_permittivity
    self.slab_permittivity = slab_permittivity
    self.input_monitor_offset = input_monitor_offset
    self.wg_min_width = wg_min_width
    self.wg_max_width = wg_max_width
    self.wg_min_separation = wg_min_separation
    self.wg_mode_padding = wg_mode_padding
    self.wg_mode_orders = wg_mode_orders
    self.left_wg_specs = left_wg_specs
    self.right_wg_specs = right_wg_specs
    self.top_wg_specs = top_wg_specs
    self.bottom_wg_specs = bottom_wg_specs
    self.wg_field_spacing_nm = wg_field_spacing_nm
    self.input_port_idx = input_port_idx

  def __post_init__(self):
    assert self.wg_mode_padding <= self.wg_min_separation


  def extent_ij(self, resolution: Q) -> Tuple[Q, Q]:
    """The total in-plane extent of the structure.

    Args:
      resolution: The resolution of the simulation.

    Returns:
      The total in-plane extent of the structure, as a tuple (i, j) of
        positions.
    """
    vi, vj = self.variable_region_size
    pml_thickness = self.pml_width * resolution
    extent_i = 2 * pml_thickness + 2 * self.wg_length + vi
    extent_j = 2 * pml_thickness + 2 * self.wg_length + vj
    return (extent_i, extent_j)

@gin.configurable
class GeneralBlockModel(model_base.Model):
  """A planar waveguide mode converter with one design region, in ceviche."""

  def __init__(
      self,
      params: _params.CevicheSimParams,
      spec: GeneralBlockSpec,
      _backend: str,
      save_wg_io_fields: bool
  ):
    """Initializes a new waveguide mode converter model.

    See the module docstring in spec.py for more details on the specification
    of the waveguide mode converter model.

    Args:
      params: Parameters for the ceviche simulation.
      spec: Specification of the waveguide mode converter geometry.
    """
    self.params = params
    self.spec = spec
    self._backend = _backend
    
    extent_i, extent_j = spec.extent_ij(params.resolution)
    self._shape = (
        u.resolve(extent_i, params.resolution),
        u.resolve(extent_j, params.resolution),
    )
    
    self._density_bg = None
    self._ports = []
    self._input_wg_field_mask = np.zeros(self._shape, dtype=np.bool_)
    self._output_wg_field_mask = np.zeros(self._shape, dtype=np.bool_)

    self._make_bg_density_and_ports()

    self.save_wg_io_fields = save_wg_io_fields

    self.density_input_only = None
    self.input_ezs = None
    self.input_hxs = None
    self.input_hys = None
    if self.save_wg_io_fields:
      self._simulate_input_port_only()
    
    self.last_forward_E = {}
    self.last_adjoint_E = {}
  
  def init_DDM_workers(self):
    # init solvers on each GPU
    gpu_ids = list(range(torch.cuda.device_count()))
    self.num_gpus = len(gpu_ids)
    self.task_queues = [mp.Queue() for _ in range(self.num_gpus)]
    self.result_queue = mp.Queue()
    self.init_queues = [mp.Queue() for _ in range(self.num_gpus)] # for passing back values after init
    
    self.processes = []

    init_kwargs = {
      'Nx': self.shape[0],
      'Ny': self.shape[1],
      'save_intermediate': False,
      'output_dir': None,
    }

    for device_id in gpu_ids:
      p = mp.Process(target=solver_worker, args=(device_id, init_kwargs, self.task_queues[device_id], self.result_queue, self.init_queues[device_id]))
      p.start()
      self.processes.append(p)
    
    init_outputs = [q.get() for q in self.init_queues]
    # check if all init outputs are the same
    assert len(set(init_outputs)) == 1
    self.source_mult = init_outputs[0]

    self.task_id_counter = 0
    self.results = {}

    self.results_lock = threading.Lock()
    self.results_cond = threading.Condition(self.results_lock)
    self.listener_thread = threading.Thread(target=self._result_listener, daemon=True)
    self.listener_thread.start()
  
  def _result_listener(self):
    while True:
      item = self.result_queue.get()
      if item is None:
        break
      task_id, result = item
      with self.results_cond:
        self.results[task_id] = result
        self.results_cond.notify_all()
  
  def stop_workers(self):
    if self._backend == 'DDM':
      for i in range(self.num_gpus):
        self.task_queues[i].put(None)
      for p in self.processes:
        p.join()

      self.result_queue.put(None)
      self.listener_thread.join()
    print("all process and threads stopped")

  def _make_bg_density_and_ports(self, init_design_region: bool = False):
    """Initializes background density and ports for the model.

    Args:
      init_design_region: `bool` specifying whether the pixels in the background
        density distribution that lie within the design region should be
        initialized to a non-zero value. If `True`, the pixels are initialized
        to a value of `1.0`.
    Side effects: Initializes `_density_bg`, an `np.ndarray` specifying the
      background material density distribution of the device. Initalizes
      `ports`, a `List[Port]` that specifies the ports of the device.
    """
    self._density_bg = None
    self._ports = []

    p = self.params
    s = self.spec
    wg_extent = s.pml_width + u.resolve(s.wg_length, p.resolution)

    def make_wgs_and_ports_for_one_side(specs, density, loc='l'):
      wg_field_spacing = u.resolve(s.wg_field_spacing_nm, p.resolution)
      if len(specs)>0:
        # The first port on the left side will be the excitation port (with largest y coord), so all ports have coordinates sorted from max to min
        previous_front = self.design_region_coords[3] if loc in ['l', 'r'] else self.design_region_coords[2]
        for idx, wg_spec in enumerate(specs):
          # create the waveguide
          center_offset, width = wg_spec
          center = s.extent_ij(p.resolution)[1]/2 + center_offset if loc in ['l', 'r'] else s.extent_ij(p.resolution)[0]/2 + center_offset
          assert width>=s.wg_min_width and width <= s.wg_max_width
          wg_min = u.resolve(center - width / 2, p.resolution)
          wg_max = u.resolve(center + width / 2, p.resolution)
          if loc == 'l':
            density[:wg_extent, wg_min:wg_max] = 1.0
            x_loc = s.pml_width + u.resolve(s.port_pml_offset, p.resolution)
            y_loc = u.resolve(center, p.resolution)
            direction = defs.Direction.X_POS
            mask_xmin, mask_xmax, mask_ymin, mask_ymax = 0, wg_extent-wg_field_spacing, wg_min-wg_field_spacing, wg_max+wg_field_spacing
          elif loc == 'r':
            density[-wg_extent:, wg_min:wg_max] = 1.0 # TO DO: check range, whether "[-wg_extent:, ]", or "[-wg_extent-1:, ]"
            x_loc = u.resolve(s.extent_ij(p.resolution)[0] - s.port_pml_offset, p.resolution) -s.pml_width
            y_loc = u.resolve(center, p.resolution)
            direction = defs.Direction.X_NEG
            mask_xmin, mask_xmax, mask_ymin, mask_ymax = self._shape[0]-wg_extent+wg_field_spacing, self._shape[0], wg_min-wg_field_spacing, wg_max+wg_field_spacing
          elif loc == 't':
            density[wg_min:wg_max, -wg_extent:] = 1.0 # TO DO: check range, whether "[ , -wg_extent:]", or "[ , -wg_extent-1:]"
            x_loc = u.resolve(center, p.resolution)
            y_loc = u.resolve(s.extent_ij(p.resolution)[1] - s.port_pml_offset, p.resolution) -s.pml_width
            direction = defs.Direction.Y_NEG
            mask_xmin, mask_xmax, mask_ymin, mask_ymax = wg_min-wg_field_spacing, wg_max+wg_field_spacing, self._shape[1]-wg_extent+wg_field_spacing, self._shape[1]
          elif loc == 'b':
            density[wg_min:wg_max, :wg_extent] = 1.0 
            x_loc = u.resolve(center, p.resolution)
            y_loc = s.pml_width + u.resolve(s.port_pml_offset, p.resolution)
            direction = defs.Direction.Y_POS
            mask_xmin, mask_xmax, mask_ymin, mask_ymax = wg_min-wg_field_spacing, wg_max+wg_field_spacing, 0, wg_extent-wg_field_spacing
          else:
            raise ValueError("loc needs to be in 'tblr'")
          assert wg_max<=previous_front # check waveguide spacing, and also within design region extent
          previous_front = wg_min - u.resolve(s.wg_min_separation, p.resolution)

          ## create the port
          self._ports.append(
            modes.WaveguidePort(
              x=x_loc,
              y=y_loc,
              width=u.resolve(width + 2 * s.wg_mode_padding,
                              p.resolution),
              order=s.wg_mode_orders[len(self._ports)],
              dir=direction,
              offset=u.resolve(s.input_monitor_offset, p.resolution) # constant doesn't depend no location
            )
          )
          ## update the wg_io field masks:
          self._output_wg_field_mask[mask_xmin:mask_xmax, mask_ymin:mask_ymax] = 1
          if len(self._ports)-1 == s.input_port_idx:
            self._input_wg_field_mask[mask_xmin:mask_xmax, mask_ymin:mask_ymax] = 1

        assert wg_min >= self.design_region_coords[1] if loc in ['l', 'r'] else self.design_region_coords[0]

    density = np.zeros(self.shape)
    make_wgs_and_ports_for_one_side(s.left_wg_specs, density, loc='l')
    make_wgs_and_ports_for_one_side(s.top_wg_specs, density, loc='t')
    make_wgs_and_ports_for_one_side(s.right_wg_specs, density, loc='r')
    make_wgs_and_ports_for_one_side(s.bottom_wg_specs, density, loc='b')

    if init_design_region:
      density[self.design_region] = 1.0

    self._density_bg = density
  
  def get_input_wg_spec_and_loc(self):
    s = self.spec
    if s.input_port_idx <len(s.left_wg_specs):
      return s.left_wg_specs[s.input_port_idx], 'l'
    elif s.input_port_idx-len(s.left_wg_specs) < len(s.top_wg_specs):
      return s.top_wg_specs[s.input_port_idx-len(s.left_wg_specs)], 't'
    elif s.input_port_idx-len(s.left_wg_specs)-len(s.top_wg_specs)<len(s.right_wg_specs):
      return s.right_wg_specs[s.input_port_idx-len(s.left_wg_specs)-len(s.top_wg_specs)], 'r'
    else:
      return s.bottom_wg_specs[s.input_port_idx-len(s.left_wg_specs)-len(s.top_wg_specs)-len(s.right_wg_specs)], 'b'

  def _simulate_input_port_only(self):
    p = self.params
    s = self.spec

    # create a density map with only the input wg, without design region and output wgs
    density_input_only = np.zeros(self.shape)

    input_wg_spec, loc = self.get_input_wg_spec_and_loc()

    center_offset, width = input_wg_spec
    center = s.extent_ij(p.resolution)[1]/2 + center_offset if loc in ['l', 'r'] else s.extent_ij(p.resolution)[0]/2 + center_offset
    wg_min = u.resolve(center - width / 2, p.resolution)
    wg_max = u.resolve(center + width / 2, p.resolution)

    if loc in ['l', 'r']:
      density_input_only[:, wg_min:wg_max] = 1.0
    else:
      density_input_only[wg_min:wg_max, :] = 1.0

    self.density_input_only = density_input_only

    # for each wavelength, simulate the response of density_input_only with the input (self._port[0])
    omegas = 2 * np.pi * u.c.to_value('nm/s') / np.asarray(self.params.wavelengths)
    pml_width = self.pml_width
    dl = self.dl
    eps_input_only = self._epsilon_r(self.density_input_only)
    flat_omegas = list(omegas.ravel(order='C'))
    num_omegas = len(flat_omegas)
    hxs, hys, ezs = [None] * num_omegas, [None] * num_omegas, [None] * num_omegas

    excite_port_idx = s.input_port_idx

    def _simulate(omega):
      sim = ceviche.fdfd_ez(
          omega,
          dl,
          eps_input_only,
          [pml_width, pml_width],
      )
      source = self.ports[excite_port_idx].source_fdfd(
          omega,
          dl,
          eps_input_only,
      )
      hx, hy, ez = sim.solve(source)
      return omega, hx, hy, ez
    
    num_workers = num_omegas
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
      simute_results = list(executor.map(_simulate,flat_omegas))
    for omega, hx, hy, ez in simute_results:
      omega_idx = flat_omegas.index(omega)
      hxs[omega_idx] = hx
      hys[omega_idx] = hy
      ezs[omega_idx] = ez
    
    self.input_hxs = np.asarray(hxs)
    self.input_hys = np.asarray(hys)
    self.input_ezs = np.asarray(ezs)
  
  def get_sources(self):
    s = self.spec
    omegas = 2 * np.pi * u.c.to_value('nm/s') / np.asarray(self.params.wavelengths)
    sources = []
    for omega in omegas:
      excite_port_idx = s.input_port_idx
      source = self.ports[excite_port_idx].source_fdfd(omega, self.dl, self.epsilon_r_bg())
      sources.append(source)
    return np.asarray(sources)
  
  def draw_output_port(self, port_idx, omega, pml_exp_decay=6, input_port_idx=0):
    # use the mode profile to draw an approximate output field, which is not accurate (in the PML region)
    p = self.params
    s = self.spec

    port = self._ports[port_idx]

    pml_width = self.pml_width
    dl = self.dl
    wg_extent = s.pml_width + u.resolve(s.wg_length, p.resolution)

    coords = port.coords()
    eps_slice = self.epsilon_r_bg()[coords]

    efield = np.zeros(self.shape, dtype=complex)
    hfield = np.zeros(self.shape, dtype=complex)
    e, h, k = port.field_profiles(eps_slice, omega, dl)
    if port.dir.is_along_x:
      start_x, end_x = max(coords[0][0] - wg_extent, 0), min(coords[0][0] + wg_extent, self.shape[0]-1)
      for i in range(start_x, end_x+1):
        pml_scale = np.exp(-pml_exp_decay*max([pml_width-i, i-(self.shape[0]-1-pml_width), 0]) / pml_width)
        e_prop = e * np.exp(1j*port.dir.sign*k*dl * (i-coords[0][0]))
        h_prop = h * np.exp(1j*port.dir.sign*k*dl * (i-coords[0][0]))
        new_coords = (i, coords[1])
        efield[new_coords] = e_prop * pml_scale
        hfield[new_coords] = port.dir.sign*h_prop * pml_scale
    else:
      start_y, end_y = max(coords[1][0] - wg_extent, 0), min(coords[1][0] + wg_extent, self.shape[1]-1)
      for i in range(start_y, end_y+1):
        pml_scale = np.exp(-pml_exp_decay*max([pml_width-i, i-(self.shape[1]-1-pml_width), 0]) / pml_width)
        e_prop = e * np.exp(1j*port.dir.sign*k*dl * (i-coords[1][0]))
        h_prop = h * np.exp(1j*port.dir.sign*k*dl * (i-coords[1][0]))
        new_coords = (coords[0], i)
        efield[new_coords] = e_prop * pml_scale
        hfield[new_coords] = -port.dir.sign*h_prop * pml_scale
      
    input_amplitude, a = calculate_amplitudes_cc(omega, dl, self._ports[input_port_idx], self.input_ezs[0], self.input_hys[0], self.input_hxs[0], self.epsilon_r_bg())
    b, output_amplitude = calculate_amplitudes_cc(omega, dl, port, efield, hfield if port.dir.is_along_x else 0., 0. if port.dir.is_along_x else hfield, self.epsilon_r_bg())

    efield = efield * self._output_wg_field_mask
    
    return efield * np.abs(input_amplitude) / np.abs(output_amplitude)
  
  def make_target_output_field(self, max_transmission, phase=0):
    assert len(self.params.wavelengths) == 1, "multi-wavelength is not supported yet"
    max_transmission = max_transmission[0,0]
    omega = 2 * np.pi * u.c.to_value('nm/s') / self.params.wavelengths[0].to_value('nm')

    target_field = np.zeros_like(self.density_bg, dtype=complex)
    for idx, t in enumerate(max_transmission):
      # TO DO: add phase reference point! Currently there is no global reference for phase
      f = self.draw_output_port(idx, omega) * np.exp(1j*phase)
      target_field += t * f
    return target_field
  
  def prepare_io_fields(self, total_efields) -> Tuple[np.ndarray, np.ndarray]:
    input_fields_wg = np.zeros_like(self.input_ezs)
    input_fields_wg = self.input_ezs*self._input_wg_field_mask[None].copy()
    output_fields_wg = total_efields*self._output_wg_field_mask[None] - input_fields_wg # subtract the input fields in the input waveguide
    output_fields_rest = total_efields*(1-self._output_wg_field_mask[None])

    return input_fields_wg, output_fields_wg, output_fields_rest  

  def update_bg_and_ports_from_given_bg_density(self, bg_density):
    assert bg_density.shape == self._shape, f"given bg_density.shape {bg_density.shape} different from model._shape {self._shape}"
    self._density_bg = bg_density
    self._ports = []

    p = self.params
    s = self.spec

    # only construct input_port for now
    x_loc, y_loc, width, direction = self.get_ports_on_one_side(bg_density, loc='l')

    self._ports.append(
      modes.WaveguidePort(
        x=x_loc,
        y=y_loc,
        width=u.resolve(width + 2 * s.wg_mode_padding, p.resolution),
        order=1, # TO DO: for now assume all order to be 1st order, change it later?
        dir=direction,
        offset=u.resolve(s.input_monitor_offset, p.resolution) # constant doesn't depend no location
      )
    )

  def get_ports_on_one_side(self, bg_density, loc='l'):
    p = self.params
    s = self.spec
    wg_extent = s.pml_width + u.resolve(s.wg_length, p.resolution)

    if loc=='l':
      density_vec = bg_density[wg_extent//2, :]
      direction = defs.Direction.X_POS
    elif loc=='r':
      density_vec = bg_density[-wg_extent//2, :]
      direction = defs.Direction.X_NEG
    elif loc=='t':
      density_vec = bg_density[:, -wg_extent//2]
      direction = defs.Direction.Y_NEG
    elif loc=='b':
      density_vec = bg_density[:, wg_extent//2]
      direction = defs.Direction.Y_POS
    else:
      raise ValueError(f"invalid loc {loc}")

    start_idx_list, end_idx_list = get_ridges(density_vec, ridge_value=1)
    start_idx, end_idx = list(zip(start_idx_list, end_idx_list))[-1]

    width = (end_idx - start_idx) * p.resolution
    if loc=='l':
      x_loc = s.pml_width + u.resolve(s.port_pml_offset, p.resolution)
      y_loc = (start_idx + end_idx)/2
    elif loc == 'r':
      x_loc = u.resolve(s.extent_ij(p.resolution)[0] - s.port_pml_offset, p.resolution) -s.pml_width
      y_loc = (start_idx + end_idx)/2
    elif loc == 't':
      x_loc = (start_idx + end_idx)/2
      y_loc = u.resolve(s.extent_ij(p.resolution)[1] - s.port_pml_offset, p.resolution) -s.pml_width
    else:
      x_loc = (start_idx + end_idx)/2
      y_loc = s.pml_width + u.resolve(s.port_pml_offset, p.resolution)
    return x_loc, y_loc, width, direction

  @property
  def design_region_coords(self) -> Tuple[int, int, int, int]:
    """The coordinates of the design region as (x_min, y_min, x_max, y_max)."""
    s = self.spec
    p = self.params
    x_min = s.pml_width + u.resolve(s.wg_length, p.resolution)
    x_max = x_min + u.resolve(s.variable_region_size[0], p.resolution)
    y_min = s.pml_width + u.resolve(s.wg_length, p.resolution)
    y_max = y_min + u.resolve(s.variable_region_size[1], p.resolution)
    return (x_min, y_min, x_max, y_max)

  @property
  def shape(self) -> Tuple[int, int]:
    """Shape of the simulation domain, in grid units."""
    return self._shape

  @property
  def density_bg(self) -> np.ndarray:
    """The background density distribution of the model."""
    return self._density_bg

  @property
  def slab_permittivity(self) -> float:
    """The slab permittivity of the model."""
    s = self.spec
    return s.slab_permittivity

  @property
  def cladding_permittivity(self) -> float:
    """The cladding permittivity of the model."""
    s = self.spec
    return s.cladding_permittivity

  @property
  def dl(self) -> float:
    """The grid resolution of the model."""
    p = self.params
    return p.resolution.to_value('m')

  @property
  def pml_width(self) -> int:
    """The width of the PML region, in grid units."""
    s = self.spec
    return s.pml_width

  @property
  def ports(self) -> List[modes.Port]:
    """A list of the device ports."""
    return self._ports

  @property
  def output_wavelengths(self) -> List[float]:
    """A list of the wavelengths, in nm, to output fields and s-parameters."""
    return u.Array(self.params.wavelengths).to_value(u.nm)
  
  def simulate(
    self,
    design_variable: np.ndarray,
    excite_port_idxs: Sequence[int] = (0,),
    wavelengths_nm: Optional[np.ndarray] = None,
    max_parallelizm: Optional[int] = None,
  ):
    if self._backend == 'ceviche':
      wavelengths_nm = np.asarray(wavelengths_nm)
      omegas = _wavelengths_nm_to_omegas(wavelengths_nm)
      return super().simulate(design_variable, excite_port_idxs, wavelengths_nm, max_parallelizm)

    assert self._backend == 'DDM'
    # setup solver using DDM
    if np.max(excite_port_idxs) > len(self.ports) - 1:
      raise ValueError('Invalid port index, {}, which exceeds the number of '
                      'ports in the device, {}.'.format(
                          np.max(excite_port_idxs),
                          len(self.ports),
                      ))
    if np.min(excite_port_idxs) < 0:
      raise ValueError('Invalid port index, {}, which below the minimum port '
                      'index of 0.'.format(np.min(excite_port_idxs),))
    if len(np.unique(excite_port_idxs)) != len(excite_port_idxs):
      raise ValueError('Duplicate port index specified in `excite_port_idxs`.')
    if not np.all(np.sort(excite_port_idxs) == np.asarray(excite_port_idxs)):
      raise ValueError('Ports specified in `excite_port_idxs` are not sorted.')

    if wavelengths_nm is None:
      wavelengths_nm = self.output_wavelengths
    else:
      wavelengths_nm = np.asarray(wavelengths_nm)
      if wavelengths_nm.ndim != 1:
        raise ValueError('`wavelengths_nm` arg must be rank-1.')

    omegas = _wavelengths_nm_to_omegas(wavelengths_nm)
    pml_width = self.pml_width
    dl = self.dl
    epsilon_r = self.epsilon_r(design_variable)
    epsilon_r_bg = self.epsilon_r_bg()

    num_excite_ports = len(excite_port_idxs)
    flat_omegas = list(omegas.ravel(order='C'))
    num_omegas = len(flat_omegas)
    sparams = [[None] * num_omegas for _ in range(num_excite_ports)]
    efields = [[None] * num_omegas for _ in range(num_excite_ports)]

    def _simulate(excite_port_idx_and_omega):
      nonlocal flat_omegas
      excite_port_idx, omega = excite_port_idx_and_omega
      source = self.ports[excite_port_idx].source_fdfd(
          omega,
          dl,
          epsilon_r_bg,
      )

      omega_torch = torch.tensor([omega], dtype=torch.float32)
      wl_torch = (2 * np.pi * C_0) / omega_torch
      dl_torch = torch.tensor([dl], dtype=torch.float32)
      epsilon_r_torch = torch.from_numpy(epsilon_r).to(torch.float32)
      source_torch = torch.from_numpy(source).to(torch.complex64)

      # send task to GPU worker queue
      task_id = self.task_id_counter
      self.task_id_counter += 1
      # device_id = task_id % self.num_gpus
      device_id = flat_omegas.index(omega) # each device handles one omega, to reuse the precomputed PML 
      last_E = self.last_forward_E.get(omega, None)
      self.task_queues[device_id].put((task_id, (epsilon_r_torch, source_torch, wl_torch, dl_torch, self.spec.pml_width, None, last_E)))

      # wait and fetch the result
      with self.results_cond:
        while task_id not in self.results:
          self.results_cond.wait()
        solution = self.results.pop(task_id)[None]

      # actually hy in ceviche convention
      hx = Ez_to_Hx(solution, dl_torch, omega_torch, torch.ones_like(solution), torch.ones_like(solution), EPSILON_0, periodic=True)
      # actually hx in ceviche convention
      hy = Ez_to_Hy(solution, dl_torch, omega_torch, torch.ones_like(solution), torch.ones_like(solution), EPSILON_0, periodic=True)
      ez = solution

      sm = []
      sp = []
      for j, port in enumerate(self.ports):
        a, b = calculate_amplitudes_torch(
            omega,
            dl,
            port,
            ez[0],
            hx[0], # actually hy in ceviche convention
            hy[0], # actually hx in ceviche convention
            epsilon_r_bg,
        )
        if j == excite_port_idx:
          sp = a
        sm.append(b)
      return excite_port_idx, omega, [smi / sp for smi in sm], ez[0]
    
    # map jobs to all available GPUs
    tasks = []
    for excite_port_idx, omega in itertools.product(excite_port_idxs, flat_omegas):
      tasks.append((excite_port_idx, omega))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
      simute_results = list(executor.map(_simulate, tasks))

    for port, omega, sps, ez in simute_results:
      port_idx = excite_port_idxs.index(port)
      omega_idx = flat_omegas.index(omega)
      sparams[port_idx][omega_idx] = sps
      efields[port_idx][omega_idx] = ez

      self.last_forward_E[omega] = ez

    # Stack into arrays and reshape for output
    sparams = np.asarray(sparams).transpose(1, 0, 2)
    efields = np.asarray(efields).transpose(1, 0, 2, 3)
    
    return sparams, efields

  def simulate_adjoint(self,
      design_variable: np.ndarray,
      excite_port_idxs: Sequence[int],
      wavelengths_nm: Optional[np.ndarray],
      max_parallelizm: Optional[int],
      forward_output_torch: Tuple[torch.Tensor],
      grad_output_torch: Tuple[torch.Tensor],
    ):
    assert self._backend == 'DDM'
    # input: args to the forward simulation, as well as grad_output_torch,
    # which is the gradient of the output from certain loss functions (same shape as the s_param, which is # of wavelengths * # of ports)

    wavelengths_nm = np.asarray(wavelengths_nm)
    omegas = _wavelengths_nm_to_omegas(wavelengths_nm)
    pml_width = self.pml_width
    dl = self.dl
    epsilon_r = self.epsilon_r(design_variable)
    epsilon_r_torch = torch.from_numpy(epsilon_r).to(torch.float32)
    epsilon_r_bg = self.epsilon_r_bg()

    num_excite_ports = len(excite_port_idxs)
    flat_omegas = list(omegas.ravel(order='C'))
    num_omegas = len(flat_omegas)

    def _adjoint_simulate(forward_ez, excite_port_idx_and_omega, grad_output):
      excite_port_idx, omega = excite_port_idx_and_omega
      omega_torch = torch.tensor([omega], dtype=torch.float32)
      dl_torch = torch.tensor([dl], dtype=torch.float32)

      # compute s-parameters on cpu
      def comp_graph(ez):
        ez = ez[None] # add batch dimension for H fields computation
        # actually hy in ceviche convention
        hx = Ez_to_Hx(ez, dl_torch, omega_torch, torch.ones_like(ez), torch.ones_like(ez), EPSILON_0, periodic=True)
        # actually hx in ceviche convention
        hy = Ez_to_Hy(ez, dl_torch, omega_torch, torch.ones_like(ez), torch.ones_like(ez), EPSILON_0, periodic=True)

        sm = []
        for j, port in enumerate(self.ports):
          a, b = calculate_amplitudes_torch(
              omega,
              dl,
              port,
              ez[0],
              hx[0], # actually hy in ceviche convention
              hy[0], # actually hx in ceviche convention
              epsilon_r_bg,
          )
          if j == excite_port_idx:
            sp = a
          sm.append(b)
        return [smi / sp for smi in sm]

      forward_ez = forward_ez.detach().requires_grad_(True)
      s_param = comp_graph(forward_ez)
      grad_ez = torch.autograd.grad(torch.stack(s_param), forward_ez, grad_outputs=torch.conj(grad_output))[0]

      # compute adjoint simulations on gpu
      wl_torch = (2 * np.pi * C_0) / omega_torch
      source_torch = torch.conj(grad_ez).to(torch.complex64).resolve_conj()  # adjoint source

      # send task to GPU worker queue
      task_id = self.task_id_counter
      self.task_id_counter += 1
      # device_id = task_id % self.num_gpus
      device_id = flat_omegas.index(omega) # each device handles one omega, to reuse the precomputed PML 
      last_E = self.last_adjoint_E.get(omega, None)
      self.task_queues[device_id].put((task_id, (epsilon_r_torch, source_torch, wl_torch, dl_torch, self.spec.pml_width, None, last_E)))
    
      # wait and fetch the result
      with self.results_cond:
        while task_id not in self.results:
          self.results_cond.wait()
        adjoint_solution = self.results.pop(task_id)
      
      self.last_adjoint_E[omega] = adjoint_solution
      
      # F(x, y) = b - A(x)* y
      # compute ∂F/∂x
      design_variable_torch = design_variable.clone().detach().requires_grad_(True)
      epsilon_for_residual = make_torch_epsilon_r(design_variable_torch, self.density_bg, self.cladding_permittivity, self.slab_permittivity, self.design_region_coords)
      sx, sy = adjoint_solution.shape # bs == 1
      bs = 1 # batch dimension for residual computation
      top_bc, bottom_bc, left_bc, right_bc = torch.zeros(bs, 1, sy, 2), torch.zeros(bs, 1, sy, 2), torch.zeros(bs, sx, 1, 2), torch.zeros(bs, sx, 1, 2)
      Sx_f, Sx_b = torch.zeros(bs, sx, sy), torch.zeros(bs, sx, sy)
      Sy_f, Sy_b = torch.zeros(bs, sy, sx), torch.zeros(bs, sy, sx)
      forward_ez = forward_ez.detach()

      forward_source = self.ports[excite_port_idx].source_fdfd(
          omega,
          dl,
          epsilon_r_bg,
      )
      forward_source_torch = torch.from_numpy(forward_source).to(torch.complex64)

      # rescale_factor = self.source_mult / source_scale(wl_torch, dl_torch)
      mult_in_res_fn = (dl_torch/wl_torch)**0.5 * MU_0/EPSILON_0*(wl_torch/(2*np.pi*dl_torch))**2
      rescale_factor = 1 / (mult_in_res_fn*source_scale(wl_torch, dl_torch))

      residual = rescale_factor * torch.view_as_complex(maxwell_robin_residue(
                                        torch.view_as_real(forward_ez[None]), 
                                        epsilon_for_residual[None], 
                                        top_bc, bottom_bc, left_bc, right_bc, 
                                        torch.view_as_real(source_scale(wl_torch, dl_torch) * forward_source_torch[None]), 
                                        (Sx_f, Sx_b), (Sy_f, Sy_b), 
                                        dl_torch, wl_torch, bc_mult=1)) # bc not in grad calculation so set to 1

      # for debugging purpose:
      # residual_adjoint = rescale_factor * torch.view_as_complex(maxwell_robin_residue(
      #                                   torch.view_as_real(adjoint_solution[None]), 
      #                                   epsilon_for_residual[None], 
      #                                   top_bc, bottom_bc, left_bc, right_bc, 
      #                                   torch.view_as_real(source_scale(wl_torch, dl_torch) * source_torch[None]), 
      #                                   (Sx_f, Sx_b), (Sy_f, Sy_b), 
      #                                   dl_torch, wl_torch, bc_mult=1))

      input_grad = torch.autograd.grad(residual[0], design_variable_torch, grad_outputs=torch.conj(adjoint_solution))[0]
      return input_grad

    input_grads = []
    forward_ezs = forward_output_torch[1] # shape (num_wavelengths, num_excite_ports, height, width)
    grad_output_s_param = grad_output_torch[0] # shape (num_wavelengths, num_excite_ports, num_ports)
    # map jobs to all available GPUs
    def worker(args):
      excite_port_idx, omega = args
      port_idx = excite_port_idxs.index(excite_port_idx)
      omega_idx = flat_omegas.index(omega)

      return _adjoint_simulate(
          forward_ezs[omega_idx, port_idx],
          (excite_port_idx, omega),
          grad_output_s_param[omega_idx, port_idx],
      )

    tasks = []
    for excite_port_idx, omega in itertools.product(excite_port_idxs, flat_omegas):
      tasks.append((excite_port_idx, omega))

    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
      input_grads = list(executor.map(worker, tasks))

    return sum(input_grads), None, None, None