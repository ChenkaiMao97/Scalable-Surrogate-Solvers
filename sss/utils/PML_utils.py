from ceviche.constants import EPSILON_0, ETA_0
import numpy as np

def make_Sx_Sy(omega, dL, Nx, Nx_pml, Ny, Ny_pml, _dir="f", float_bits=32):
    dtype = np.complex64 if (float_bits==32) else np.complex128
    if _dir == 'f':
        sfactor_x = create_sfactor_f(omega, dL, Nx, Nx_pml, float_bits=float_bits)
        sfactor_y = create_sfactor_f(omega, dL, Ny, Ny_pml, float_bits=float_bits)
    elif _dir == 'b':
        sfactor_x = create_sfactor_b(omega, dL, Nx, Nx_pml, float_bits=float_bits)
        sfactor_y = create_sfactor_b(omega, dL, Ny, Ny_pml, float_bits=float_bits)

    Sx_2D = np.zeros((Nx,Ny), dtype=dtype)
    Sy_2D = np.zeros((Nx,Ny), dtype=dtype)

    for i in range(0, Ny):
        Sx_2D[:, i] = sfactor_x
    for i in range(0, Nx):
        Sy_2D[i, :] = sfactor_y

    return Sx_2D, Sy_2D


def create_sfactor_f(omega, dL, N, N_pml, float_bits=None):
    # forward
    dtype = np.complex64 if (float_bits==32) else np.complex128
    sfactor_array = np.ones(N, dtype=dtype)

    if N_pml == 0:
        return sfactor_array

    dw = N_pml*dL
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 0.5), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 0.5), dw, omega)

    return sfactor_array


def create_sfactor_b(omega, dL, N, N_pml, float_bits=None):
    # backward
    dtype = np.complex64 if (float_bits==32) else np.complex128
    sfactor_array = np.ones(N, dtype=dtype)
    
    if N_pml == 0:
        return sfactor_array

    dw = N_pml*dL
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 1), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 1), dw, omega)

    return sfactor_array

def sig_w(l, dw, m=3, lnR=-30):
    """ Fictional conductivity, note that these values might need tuning """
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw)**m

def s_value(l, dw, omega):
    """ S-value to use in the S-matrices """
    # l is distance to the boundary of pml (close to the center)
    # dw is the physical thickness of pml (N_pml * dL)

    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0)