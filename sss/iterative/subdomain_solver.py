# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import os, sys
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from sss.iterative.gmres import GMRES
from sss.iterative.gmres_setup import GMRES_setup
from sss.iterative.bicgstab import BICGSTAB
from sss.models import model_factory
from sss.utils.PDE_utils import maxwell_robin_Aop, maxwell_robin_residue, maxwell_robin_damping_residue, maxwell_robin_damping_Aop
from sss.utils.plot_utils import plot_helper, setup_plot_data
from sss.utils.UI import printc

from functools import partial
from collections import OrderedDict
from typing import Callable

import gin

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

@gin.configurable
class SubdomainSolver:
    def __init__(
        self,
        trainer_fn: Callable,
        solver_type: str = "GMRES",
        residual_type: str = "UPML",
        model_fn = model_factory,
        dataset_fn = None, # if given, sample data from dataset
        output_dir: str = None,
        model_loading_path: str = None,
        num_samples: int = 1,
        seed: int = 0,
        max_iter: int = 20,
        tol: float = 1e-6,
        plot_x_r_history: bool = False,
        old_model: bool = False,
        plot_N: int = 6,
        gpu_id: int = 0,
    ):
        self.trainer_fn = trainer_fn
        self.solver_type = solver_type
        self.model_fn = model_fn
        self.dataset_fn = dataset_fn
        self.output_dir = output_dir
        self.model_loading_path = model_loading_path
        self.seed = seed
        self.num_samples = num_samples
        self.source_mult = None
        self.bc_mult = None
        self.max_iter = max_iter
        self.tol = tol
        self.plot_x_r_history = plot_x_r_history
        self.old_model = old_model
        self.plot_N = plot_N
        self.residual_type = residual_type
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        if self.residual_type == "UPML":
            self.residue_function = maxwell_robin_residue
            self.Aop = maxwell_robin_Aop
        elif self.residual_type == "damping":
            self.residue_function = maxwell_robin_damping_residue
            self.Aop = maxwell_robin_damping_Aop
        else:
            raise ValueError(f"Residual type {self.residual_type} not supported")
    def init(self):
        # set seed:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        # setup output directory:
        if self.output_dir is not None and self.plot_x_r_history:
            os.makedirs(self.output_dir, exist_ok=True)

        # load model:
        if self.old_model:
            self.load_from_old_model()
        else:
            self.load_from_DDP()

        # init iterative solver:
        if self.solver_type == "GMRES_setup":
            self.solver = GMRES_setup(model=self.model, max_iter=self.max_iter, tol=self.tol)
        elif self.solver_type == "GMRES":
            self.solver = GMRES(model=self.model, max_iter=self.max_iter, tol=self.tol)
        elif self.solver_type == "BICGSTAB":
            self.solver = BICGSTAB(model=self.model, max_iter=self.max_iter, tol=self.tol)
        else:
            raise ValueError(f"Solver type {self.solver_type} not supported")
        
        # init dataset:
        if self.dataset_fn is not None:
            self.dataset = self.dataset_fn()
            self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
    
    def load_from_DDP(self):
        # if not dist.is_initialized():
        #     setup(0,1)
        load_path = self.model_loading_path
        # parse all the gin config files in the load_path:
        for gin_file in os.listdir(load_path):
            if gin_file.endswith('.gin') and 'job_config' not in gin_file:
                print(f"Parsing gin config file {gin_file}")
                gin.parse_config_file(os.path.join(load_path, gin_file))
        dummy_trainer = self.trainer_fn(model_config=None, model_saving_path = None)
        self.source_mult = dummy_trainer.source_mult
        self.source_mult_model = 1
        self.bc_mult = dummy_trainer.bc_mult
        printc(f"source_mult: {self.source_mult}, bc_mult: {self.bc_mult}", 'r')

        printc(f"Restoring weights from {os.path.join(load_path, 'models/best_model.pt')}", 'r')
        checkpoint = torch.load(os.path.join(load_path, 'models/best_model.pt'))

        # if saved with DDP, remove 'module.' prefix:
        stripped_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            stripped_state_dict[new_key] = v

        self.model = self.model_fn()
        self.model.load_state_dict(stripped_state_dict)
        self.model.eval()
        self.model.to(self.device)
    
    def load_from_old_model(self):
        setup(0,1)
        sys.path.append("sss/models")
        printc(f"Restoring weights from {self.model_loading_path}/best_model.pt", 'r')
        checkpoint = torch.load(self.model_loading_path+"/best_model.pt", map_location=self.device)
        self.model = checkpoint['model'].module
        self.model.eval()
        self.model.to(self.device)
        self.source_mult_model = 1
        self.source_mult = 1.0
        self.bc_mult = 30.0
        
    def setup(self, args, init_x=None):
        source, Sxf, Syf, Sxb, Syb, eps, dL, wl = args
        # setup M and Aop for iterative solver:
        self.solver.setup_M(eps, source, Sxf, Sxb, Syf, Syb, wl, dL, self.source_mult_model, init_x=init_x)
        Aop = partial(self.Aop, eps=eps, source=self.source_mult*source, Sxs=(Sxf, Sxb), Sys=(Syf, Syb), dL=dL, wl=wl, bc_mult=self.bc_mult)
        Aop_complex = lambda x: torch.view_as_complex(Aop(torch.view_as_real(x)))
        self.solver.setup_Aop(Aop_complex)
        self.residue_fn = partial(self.residue_function, x=torch.zeros_like(source), eps=eps, source=self.source_mult*source, Sxs=(Sxf, Sxb), Sys=(Syf, Syb), dL=dL, wl=wl, bc_mult=self.bc_mult)

    def solve(self, args, max_iter_override=None):
        top_bc, bottom_bc, left_bc, right_bc = args
        # x = torch.zeros((top_bc.shape[0], left_bc.shape[1], top_bc.shape[2], 2), dtype=torch.float32, device=top_bc.device)
        rhs = self.residue_fn(top_bc=top_bc, bottom_bc=bottom_bc, left_bc=left_bc, right_bc=right_bc)
        rhs = torch.view_as_complex(rhs)
        solution, r, relres_history, x_history, r_history = self.solver.solve(rhs, max_iter=max_iter_override, return_xr_history=self.plot_x_r_history)

        return r, solution, relres_history, x_history, r_history

    def solve_from_dataset(self):
        assert self.dataset_fn is not None, "dataset_fn is not given"

        for i, sample_batched in enumerate(self.dataloader):
            if i >= self.num_samples:
                break
            # load data:
            y, source, Sxf, Syf, Sxb, Syb, eps, top_bc, bottom_bc, left_bc, right_bc, dL, wl = sample_batched['field'].cuda(non_blocking=True), sample_batched['source'].cuda(non_blocking=True), sample_batched['Sx_f'].cuda(non_blocking=True), sample_batched['Sy_f'].cuda(non_blocking=True), sample_batched['Sx_b'].cuda(non_blocking=True), sample_batched['Sy_b'].cuda(non_blocking=True), sample_batched['eps'].cuda(non_blocking=True), sample_batched['top_bc'].cuda(non_blocking=True), sample_batched['bottom_bc'].cuda(non_blocking=True), sample_batched['left_bc'].cuda(non_blocking=True), sample_batched['right_bc'].cuda(non_blocking=True), sample_batched['dL'].cuda(non_blocking=True), sample_batched['wl'].cuda(non_blocking=True)
            print(wl, dL)
            self.setup((source/self.source_mult, Sxf, Syf, Sxb, Syb, eps, dL, wl))
            r, solution, relres_history, x_history, r_history = self.solve((top_bc, bottom_bc, left_bc, right_bc))
            
            if self.plot_x_r_history:
                # save x_history and r_history:
                residue_gt = self.residue_function(y, eps, top_bc, bottom_bc, left_bc, right_bc, source, (Sxf, Sxb), (Syf, Syb), dL, wl, bc_mult=self.bc_mult)

                colored_setup = setup_plot_data(eps[0].detach().cpu(), source[0,...,0].detach().cpu(), sx_f=Sxf[0].detach().cpu(), sy_f=Syf[0].detach().cpu())
                plot_data = [colored_setup,self.source_mult*source[0,...,0]+source[0,...,1],Sxf[0]+Syf[0],y[0,...,0],torch.abs(residue_gt[0,...,0]),torch.abs(torch.view_as_real(r)[0,...,0])]
                plot_title = ['eps', 'source', 'pml', 'gt', 'residue_gt', 'residue_input']
                cmaps = [None, 'seismic', None, 'seismic', 'Reds', 'Reds']
                center_zero = [False, False, False, True, False, False]
                row, column = 1, 6
                plot_helper(plot_data,row,column,plot_title,self.output_dir+f"sample_{i}_residue_wl{wl[0].item():.2e}_dL{dL[0].item():.2e}.png", cmaps=cmaps, center_zero=center_zero)
                
                plot_indices = [round(x) for x in np.linspace(0, len(x_history)-1, self.plot_N).astype(int)]
                x_history = [x_history[j] for j in plot_indices]
                r_history = [r_history[j] for j in plot_indices]

                plot_data = [*[x[0].real for x in x_history], *[y[0,...,0]-x[0].real for x in x_history], *[torch.abs(r[0]) for r in r_history]]
                plot_title = [*[f"output_{j}" for j in plot_indices], "gt", *[f"error_{j}" for j in plot_indices[1:]], "init_residual", *[f"residual_{j}" for j in plot_indices[1:]]]
                cmaps = [*['seismic' for _ in range(self.plot_N)], *['seismic' for _ in range(self.plot_N)], *['Reds' for _ in range(self.plot_N)]]
                center_zero = [*[True for _ in range(self.plot_N)], *[True for _ in range(self.plot_N)], *[False for _ in range(self.plot_N)]]
                row, column = 3, self.plot_N
                plot_helper(plot_data,row,column,plot_title,self.output_dir+f"sample_{i}_debug_recurrent_wl{wl[0].item():.2e}_dL{dL[0].item():.2e}.png", cmaps=cmaps, center_zero=center_zero)

            else:
                print(f"Sample {i}: relres_history: {relres_history}")
