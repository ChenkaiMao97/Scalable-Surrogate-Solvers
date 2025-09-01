# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

# sparse solver with scipy

import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np
import torch
from ceviche.constants import C_0, MU_0, EPSILON_0

# try:
#     from pyMKL import pardisoSolver
#     HAS_MKL = True
#     print('using MKL for direct solvers')
# except:
#     HAS_MKL = False
#     print('using scipy.sparse for direct solvers.  Note: using MKL will make things significantly faster.')
HAS_MKL = False

def _solve_direct(A, b):
    """ Direct solver """
    if HAS_MKL:
        pSolve = pardisoSolver(A, mtype=13)
        pSolve.factor()
        x = pSolve.solve(b)
        pSolve.clear()
        return x
    else:
        # scipy solver, slower
        return spl.spsolve(A, b)

class SparseDirectSolver:
    def __init__(self):
        self.As = []
        self.lus = []
    
    def clear(self):
        self.As = []
        self.lus = []
    
    def setup(self, eps, Sx_f_I, Sy_f_I, Sx_b_I, Sy_b_I, dL, wl):
        """
        eps, source and S matrix are list of tensors, which may have different shapes.
        This function constructs the A matrices in Ax = b, for each subdomain.
        b will be provided when solving, as boundary values.
        """
        if len(self.As) > 0:
            return
        print("make all As")
        self.dl = dL
        self.wl = wl
        for i in range(len(eps)):
            A, lu = self.make_A(eps[i], Sx_f_I[i], Sy_f_I[i], Sx_b_I[i], Sy_b_I[i], dL, wl)
            self.As.append(A)
            self.lus.append(lu)
        print("done making As")

    def make_A(self, eps, Sx_f_I, Sy_f_I, Sx_b_I, Sy_b_I, dL, wl):
        """
        Construct the A matrix for a single subdomain.
        """

        nx, ny = eps.shape
        N = nx * ny

        def idx(i, j):
            return i * ny + j

        k0 = 2 * np.pi / wl
        omega = k0 * C_0

        # Precompute indices
        I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        P = idx(I, J)

        # Prepare storage for COO format
        row = []
        col = []
        data = []

        # Robin BC at domain boundaries
        # Top/Bottom (i = 0 or nx-1)
        for i_edge in [0, nx - 1]:
            i = i_edge
            j = np.arange(ny)
            p = idx(i, j)
            eps_ij = eps[i, j]
            sxf_ij = 1 + 1j * Sx_f_I[i, j]
            val = 1 + 0.5j * k0 * np.sqrt(eps_ij * sxf_ij) * dL
            row.extend(p)
            col.extend(p)
            data.extend(val)
            i_adj = i + 1 if i == 0 else i - 1
            p_adj = idx(i_adj, j)
            val_adj = -1 + 0.5j * k0 * np.sqrt(eps_ij * sxf_ij) * dL
            row.extend(p)
            col.extend(p_adj)
            data.extend(val_adj)

        # Left/Right (j = 0 or ny-1)
        for j_edge in [0, ny - 1]:
            j = j_edge
            i = np.arange(1,nx-1) # attention: top and bottom have priority
            p = idx(i, j)
            eps_ij = eps[i, j]
            syf_ij = 1 + 1j * Sy_f_I[i, j]
            val = 1 + 0.5j * k0 * np.sqrt(eps_ij * syf_ij) * dL
            row.extend(p)
            col.extend(p)
            data.extend(val)
            j_adj = j + 1 if j == 0 else j - 1
            p_adj = idx(i, j_adj)
            val_adj = -1 + 0.5j * k0 * np.sqrt(eps_ij * syf_ij) * dL
            row.extend(p)
            col.extend(p_adj)
            data.extend(val_adj)

        # Inner domain
        inner_mask = np.ones((nx, ny), dtype=bool)
        inner_mask[0, :] = inner_mask[-1, :] = inner_mask[:, 0] = inner_mask[:, -1] = False
        i_inner, j_inner = np.where(inner_mask)
        p = idx(i_inner, j_inner)

        eps_ij = eps[i_inner, j_inner]
        sxf_ij = 1 + 1j * Sx_f_I[i_inner, j_inner]
        syf_ij = 1 + 1j * Sy_f_I[i_inner, j_inner]
        sxb_ij = 1 + 1j * Sx_b_I[i_inner, j_inner]
        sxb1_ij = 1 + 1j * Sx_b_I[i_inner + 1, j_inner]
        syb_ij = 1 + 1j * Sy_b_I[i_inner, j_inner]
        syb1_ij = 1 + 1j * Sy_b_I[i_inner, j_inner + 1]

        k2_eps = eps_ij * (k0 * dL)**2

        A_xm = 1 / (sxf_ij * sxb_ij)
        A_xp = 1 / (sxf_ij * sxb1_ij)
        A_ym = 1 / (syf_ij * syb_ij)
        A_yp = 1 / (syf_ij * syb1_ij)
        A_c = k2_eps - (A_xm + A_xp + A_ym + A_yp)

        row.extend(p)
        col.extend(idx(i_inner - 1, j_inner))
        data.extend(A_xm)

        row.extend(p)
        col.extend(idx(i_inner + 1, j_inner))
        data.extend(A_xp)

        row.extend(p)
        col.extend(idx(i_inner, j_inner - 1))
        data.extend(A_ym)

        row.extend(p)
        col.extend(idx(i_inner, j_inner + 1))
        data.extend(A_yp)

        row.extend(p)
        col.extend(p)
        data.extend(A_c)

        A = sp.coo_matrix((np.array(data, dtype=np.complex64), (np.array(row), np.array(col))), shape=(N, N))

        # store LU decomposition
        lu = spl.splu(A.tocsc())
        return A.tocsr(), lu

    # def make_A(self, eps, Sx_f_I, Sy_f_I, Sx_b_I, Sy_b_I, dL, wl):
    #     """
    #     Construct the A matrix for a single subdomain.
    #     """

    #     nx, ny = eps.shape
    #     N = nx * ny

    #     A = sp.lil_matrix((N, N), dtype=np.complex64)

    #     def idx(i, j):
    #         return i * ny + j

    #     k0 = 2*np.pi / wl
    #     omega = k0*C_0

    #     for i in range(nx):
    #         for j in range(ny):
    #             p = idx(i, j)
    #             eps_ij = eps[i, j]
    #             sxf_ij = 1+1j*Sx_f_I[i, j]
    #             syf_ij = 1+1j*Sy_f_I[i, j]

    #             if i == 0:
    #                 A[p, p] = 1 + 1/2* 1j * k0 * (eps_ij*sxf_ij)**0.5 * dL
    #                 A[p, idx(i+1, j)] = -1 + 1/2* 1j * k0 * (eps_ij*sxf_ij)**0.5 * dL
    #                 continue
    #             if i == nx-1:
    #                 A[p, p] = 1 + 1/2* 1j * k0 * (eps_ij*sxf_ij)**0.5 * dL
    #                 A[p, idx(i-1, j)] = -1 + 1/2* 1j * k0 * (eps_ij*sxf_ij)**0.5 * dL
    #                 continue
    #             if j == 0:
    #                 A[p, p] = 1 + 1/2* 1j * k0 * (eps_ij*syf_ij)**0.5 * dL
    #                 A[p, idx(i, j+1)] = -1 + 1/2* 1j * k0 * (eps_ij*syf_ij)**0.5 * dL
    #                 continue
    #             if j == ny-1:
    #                 A[p, p] = 1 + 1/2* 1j * k0 * (eps_ij*syf_ij)**0.5 * dL
    #                 A[p, idx(i, j-1)] = -1 + 1/2* 1j * k0 * (eps_ij*syf_ij)**0.5 * dL
    #                 continue

    #             # inner pixels:
    #             k2_eps = np.complex64(eps_ij * k0**2 * dL**2)
    #             sxb_ij = 1+1j*Sx_b_I[i, j]
    #             sxb1_ij = 1+1j*Sx_b_I[i+1, j]
    #             syb_ij = 1+1j*Sy_b_I[i, j]
    #             syb1_ij = 1+1j*Sy_b_I[i, j+1]
    #             A[p, idx(i - 1, j)] = np.complex64(1 / (sxf_ij * sxb_ij))
    #             A[p, idx(i + 1, j)] = np.complex64(1 / (sxf_ij * sxb1_ij))
    #             A[p, idx(i, j - 1)] = np.complex64(1 / (syf_ij * syb_ij))
    #             A[p, idx(i, j + 1)] = np.complex64(1 / (syf_ij * syb1_ij))
    #             A[p, p] = k2_eps - (A[p, idx(i - 1, j)] + A[p, idx(i + 1, j)] + A[p, idx(i, j - 1)] + A[p, idx(i, j + 1)])
    #     return A.tocsr()
    
    def make_b(self, source, top_bc, bottom_bc, left_bc, right_bc):
        nx, ny = source.shape
        N = nx * ny
        rhs = - MU_0 / EPSILON_0 * source.numpy().flatten()

        def idx(i, j):
            return i * ny + j

        rhs[idx(0, np.arange(ny))] = np.complex64(top_bc[0, :])
        rhs[idx(nx - 1, np.arange(ny))] = np.complex64(bottom_bc[0, :])
        rhs[idx(np.arange(1,nx-1), 0)] = np.complex64(left_bc[1:-1, 0])
        rhs[idx(np.arange(1,nx-1), ny - 1)] = np.complex64(right_bc[1:-1, 0])

        return rhs
    
    def make_bs(self, source, top_bc, bottom_bc, left_bc, right_bc):
        bs = []
        for i in range(len(source)):
            b = self.make_b(source[i], top_bc[i], bottom_bc[i], left_bc[i], right_bc[i])
            bs.append(b)
        return bs
    
    def solve_all(self, source, top_bc, bottom_bc, left_bc, right_bc):
        bs = self.make_bs(source, top_bc, bottom_bc, left_bc, right_bc)

        res = []
        for i in range(len(bs)):
            out = torch.from_numpy(self.solve(self.lus[i], bs[i])).reshape(source[i].shape)
            res.append(out)
        return res

    def solve(self, lu, b):
        return lu.solve(b)
        # return _solve_direct(A, b)

        # debug checking:
        # if np.isnan(A.data).any() or np.isinf(A.data).any():
        #     print("A has NaNs or Infs")
        # if np.isnan(b).any() or np.isinf(b).any():
        #     print("b has NaNs or Infs")
        
        # print("A shape:", A.shape)
        # print("A dtype:", A.dtype)
        # print("b shape:", b.shape)
        # print("b dtype:", b.dtype)
        
        # eig_max = spl.eigs(A, k=1, which='LM', return_eigenvectors=False)[0]
        # # Smallest magnitude eigenvalue (may be unstable for singular/ill-conditioned matrices)
        # eig_min = spl.eigs(A, k=1, which='SM', return_eigenvectors=False)[0]
        # # Estimate condition number
        # cond_est = abs(eig_max) / abs(eig_min)
        # print(f"Condition number: {cond_est}")

        # diagonal = A.diagonal()
        # if np.any(diagonal == 0):
        #     print("Zero entries on diagonal – may be singular")

        # row_sums = np.array(A.sum(axis=1)).flatten()
        # if np.any(row_sums == 0):
        #     print("A has zero rows – definitely singular")

        # try:
        #     lu = spl.splu(A.tocsc())
        #     print("LU decomposition successful")
        # except RuntimeError as e:
        #     print("Matrix is singular or structurally singular:", e)

