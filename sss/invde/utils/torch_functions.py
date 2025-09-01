import torch
import numpy as np
from ceviche_challenges import defs
from ceviche_challenges import modes
from typing import Tuple

# helper functions in torch
def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Compute the cross product between two VectorFields."""
  return (
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  )

def safe_conj(a):
  if isinstance(a, torch.Tensor):
    return torch.conj(a)
  else:
    return torch.conj(torch.tensor(a))

def overlap(
    a: torch.Tensor,
    b: torch.Tensor,
    normal: defs.Direction,
) -> float:
  """Numerically compute the overlap integral of two VectorFields.

  Args:
    a: `VectorField` specifying the first field.
    b: `VectorField` specifying the second field.
    normal: `Direction` specifying the direction normal to the plane (or slice)
      where the overlap is computed.

  Returns:
    Result of the overlap integral.
  """
  ac = tuple([safe_conj(ai) for ai in a])
  return torch.sum(cross(ac, b)[normal.index])

def calculate_amplitudes(
    omega: float,
    dl: float,
    port: modes.Port,
    ez: torch.Tensor,
    hy: torch.Tensor,
    hx: torch.Tensor,
    epsilon_r: np.ndarray,
) -> Tuple[float, float]:
  """Calculate amplitudes of the forward and backward waves at a port.

  Args:
    omega: `float` specifying the angular frequency of the mode, in units of
      rad/sec.
    dl: `float` specifying the spatial grid cell size, in units of meters.
    port: `Port` specifying details of the mode calculation, e.g. order,
      location, direction, etc.
    ez: `Field` specifying the distribution of the z-component of the electric
      field
    hy: `Field` specifying the distribution of the y-component of the magnetic
      field.
    hx: `Field` specifying the distribution of the x-component of the magnetic
      field.
    epsilon_r: `Geometry` specifying the permitivitty distribution.

  Returns:
    A tuple consisting of the complex-valued scattering parameters, (s_+, s_-)
    at the port.
  """
  coords = port.coords()
  et_m, ht_m, _ = port.field_profiles(epsilon_r[coords], omega, dl)

  if port.dir.is_along_x:
    coords_offset = (coords[0] + port.signed_offset(), coords[1])
    h = (0., torch.ravel(hy[coords_offset]), 0)
    hm = (0., torch.from_numpy(ht_m), 0.)
    # The E-field is not co-located with the H-field in the Yee cell. Therefore,
    # we must sample at two neighboring pixels in the propataion direction and
    # then interpolate:
    coords_e = (coords_offset[0] + np.array([[-1], [0]]), coords_offset[1])
    e_yee_shifted = 0.5 * torch.sum(ez[coords_e], dim=0)
  else:
    coords_offset = (coords[0], coords[1] + port.signed_offset())
    h = (torch.ravel(hx[coords_offset]), 0, 0)
    hm = (-torch.from_numpy(ht_m), 0., 0.)
    # The E-field is not co-located with the H-field in the Yee cell. Therefore,
    # we must sample at two neighboring pixels in the propataion direction and
    # then interpolate:
    coords_e = (coords_offset[0], coords_offset[1] + np.array([[-1], [0]]))
    e_yee_shifted = 0.5 * torch.sum(ez[coords_e], dim=0)

  e = (0., 0., e_yee_shifted)
  em = (0., 0., torch.from_numpy(et_m))

  overlap1 = overlap(em, h, port.dir)
  overlap2 = overlap(hm, e, port.dir)
  normalization = overlap(em, hm, port.dir)

  # Phase convention in ceviche is exp(+jwt-jkz)
  if port.dir.sign > 0:
    s_p = (overlap1 + overlap2) / 2 / torch.sqrt(2 * normalization)
    s_m = (overlap1 - overlap2) / 2 / torch.sqrt(2 * normalization)
  else:
    s_p = (overlap1 - overlap2) / 2 / torch.sqrt(2 * normalization)
    s_m = (overlap1 + overlap2) / 2 / torch.sqrt(2 * normalization)

  return s_p, s_m

def make_torch_epsilon_r(
  design_variable: torch.Tensor,
  density_bg: np.ndarray,
  cladding_permittivity: float,
  slab_permittivity: float,
	design_region_coords
) -> torch.Tensor:
	# both density_bg and design_variable have value between 0 and 1
	# design_variable requires grad, while density_bg does not
	
	destination_ = torch.tensor(density_bg, dtype=torch.float32, requires_grad=False)
	destination_[design_region_coords[0]:design_region_coords[2], design_region_coords[1]:design_region_coords[3]] = design_variable
	epsilon_r = cladding_permittivity + (slab_permittivity - cladding_permittivity) * destination_

	return epsilon_r
