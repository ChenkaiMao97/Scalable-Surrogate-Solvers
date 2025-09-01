# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import  os, sys, timeit
import  numpy as np
import pandas as pd

from sss.trainers import BaseTrainer
from sss.utils.PDE_utils import maxwell_robin_residue, maxwell_robin_Aop
from sss.utils.UI import printc
from sss.models import model_factory
from sss.utils.plot_utils import plot_helper

import torch
from torch import optim
from torch.utils.data import random_split, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from fvcore.nn import FlopCountAnalysis

from tqdm import tqdm
from typing import Callable, List
import matplotlib.pyplot as plt
from time import sleep
import gin

def MAE_loss(a,b=None):
    if b is None:
        return torch.mean(torch.abs(a))
    else:
        return torch.mean(torch.abs(a-b))/torch.mean(torch.abs(b))

def MSE_loss(a,b=None):
    if b is None:
        return torch.mean(a**2)
    else:
        return torch.mean((a-b)**2)

def r2c(x):
    return torch.view_as_complex(x)

def c2r(x):
    return torch.view_as_real(x)

def complex_max(x, eps=1e-6):
    return torch.where(torch.abs(x) > eps, x, torch.tensor(eps))

@gin.configurable
class TrainerGMRES(BaseTrainer):
    def __init__(
        self, 
        model_config: str,
        domain_sizex,
        domain_sizey,
        f_padding,
        epoch, 
        batch_size,
        model_saving_path,
        dataset_fn,
        start_lr,
        end_lr,
        weight_decay,
        model_fn: Callable = model_factory,
        continue_train: bool = False,
        load_from_checkpoint_path: str = None, # if given, start from this checkpoint, i.e. transfer learning
        world_size: int = 1,
        seed: int = 42,
        GMRES_iter: int = 1,
        residual_clip: float = 100.0,
        bc_mult: float = 1.0,
        source_mult: float = 1.0,
        bloch_vector: List[float] = None,
        phys_start_epoch: int = 1,
        ratio: float = 0.1
    ):
        self.model_config = model_config

        self.model_fn = model_fn
        self.domain_sizex = domain_sizex
        self.domain_sizey = domain_sizey
        self.f_padding = f_padding
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_saving_path = model_saving_path
        self.dataset_fn = dataset_fn
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.continue_train = continue_train
        self.load_from_checkpoint_path = load_from_checkpoint_path
        self.world_size = world_size
        self.seed = seed
        self.GMRES_iter = GMRES_iter
        printc(f"training with GMRES_iter: {self.GMRES_iter}", 'g')
        self.residual_clip = residual_clip
        self.bc_mult = bc_mult
        self.source_mult = source_mult
        self.bloch_vector = bloch_vector

        self.phys_start_epoch = phys_start_epoch
        self.ratio = ratio
    
    def init(self):
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        os.makedirs(self.model_saving_path, exist_ok=True)
        os.makedirs(self.model_saving_path+'/plots', exist_ok=True)

        # setup dataloader
        self.ds = self.dataset_fn()
        self.train_ds, self.test_ds = random_split(self.ds, [int(0.9*len(self.ds)), len(self.ds) - int(0.9*len(self.ds))])

        self.residual_fn = maxwell_robin_residue
        self.Aop = maxwell_robin_Aop
    def distributed_training(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8764'
        mp.spawn(self.train, nprocs=self.world_size)
    
    def train(self, rank):
        # for each new process, parse the model config:
        gin.parse_config_files_and_bindings([self.model_config], bindings=[])
        
        torch.cuda.set_device(rank)

        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size=self.world_size,                              
            rank=rank                                               
        )

        assert self.batch_size % self.world_size == 0
        each_GPU_batch_size = self.batch_size // self.world_size

        # use start_lr and end_lr to calculate lr update steps:
        total_steps = self.epoch*len(self.train_ds)/self.batch_size
        update_times = np.log(self.end_lr/self.start_lr)/np.log(0.99)
        lr_update_steps = int(total_steps/update_times)
        if rank==0:
            printc(f"start_lr: {self.start_lr}, end_lr: {self.end_lr}, total_steps: {total_steps}, lr_update_steps: {lr_update_steps}", 'b')

        start_epoch=0
        if self.continue_train:
            load_path = self.load_from_checkpoint_path if self.load_from_checkpoint_path is not None else self.model_saving_path
            if rank==0:
                printc(f"Restoring weights from {load_path}/best_model.pt", 'r')
                if self.load_from_checkpoint_path:
                    df = pd.DataFrame(columns=['epoch', 'lr', 'train_data_loss', 'train_residual_loss', 'test_data_loss', 'test_residual_loss'])
                else:
                    df = pd.read_csv(load_path+"/df.csv")
            checkpoint = torch.load(load_path+"/best_model.pt", map_location=torch.device(f'cuda:{rank}'))
            if self.load_from_checkpoint_path: # if load from checkpoint path, start from epoch 0
                start_epoch=0
            else: # if continue training for the same job, start from the last epoch
                start_epoch=checkpoint['epoch']+1
            model = checkpoint['model'].module
            optimizer = torch.optim.Adam(model.parameters(), lr=self.start_lr, weight_decay=self.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'].state_dict())
        else:
            if rank==0:
                df = pd.DataFrame(columns=['epoch', 'lr', 'train_residual_loss', 'test_residual_loss'])
            
            model = self.model_fn(domain_sizes = (self.domain_sizex, self.domain_sizey), paddings = (self.f_padding, self.f_padding))
            optimizer = torch.optim.Adam(model.parameters(), lr=self.start_lr, weight_decay=self.weight_decay)
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

        if rank==0:
            tmp = filter(lambda x: x.requires_grad, model.parameters())
            num = sum(map(lambda x: np.prod(x.shape), tmp))
            printc(f'Total trainable tensors: {num}', 'r')
            with open(self.model_saving_path + '/'+'config.txt', 'w') as f:
                f.write(model.__str__())
                f.write(f'Total trainable tensors: {num}')
        
        model.cuda(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_ds,
            num_replicas=self.world_size,
            rank=rank,
            shuffle=True
        )

        test_sampler = torch.utils.data.distributed.DistributedSampler(
            self.test_ds,
            num_replicas=self.world_size,
            rank=rank,
            shuffle=True
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=each_GPU_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=self.ds.collate_fn_same_wl_dL
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=each_GPU_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.ds.collate_fn_same_wl_dL
        )

        FLOPs_recorded = False
        gradient_count = 0
        best_loss = 1e4
        running_residual_loss = 1.0

        for step in range(start_epoch, self.epoch):
            train_sampler.set_epoch(step)
            test_sampler.set_epoch(step)
            epoch_start_time = timeit.default_timer()
            if rank==0:
                printc(f"epoch: {step}", 'g')
                
            # training
            model.train()
            if rank==0:
                pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            else:
                pbar = enumerate(train_loader)
            for idx, sample_batched in pbar:
                gradient_count += 1
                optimizer.zero_grad(set_to_none=True)

                eps, source, dL, wl = sample_batched['eps'].cuda(non_blocking=True), sample_batched['source'].cuda(non_blocking=True), sample_batched['dL'].cuda(non_blocking=True), sample_batched['wl'].cuda(non_blocking=True)
                top_bc, bottom_bc, left_bc, right_bc = sample_batched['top_bc'].cuda(non_blocking=True), sample_batched['bottom_bc'].cuda(non_blocking=True), sample_batched['left_bc'].cuda(non_blocking=True), sample_batched['right_bc'].cuda(non_blocking=True)
                Sxf, Sxb, Syf, Syb = sample_batched['sx_f'].cuda(non_blocking=True), sample_batched['sx_b'].cuda(non_blocking=True), sample_batched['sy_f'].cuda(non_blocking=True), sample_batched['sy_b'].cuda(non_blocking=True)                

                init_x = 1e-3*torch.randn_like(source, dtype=torch.float32)
                init_residual = self.residual_fn(init_x, eps[...,0], top_bc, bottom_bc, left_bc, right_bc, self.source_mult*source, (Sxf, Sxb), (Syf,Syb), dL, wl, self.bc_mult, clamp=self.residual_clip)

                # record FLOPs for once:
                # if not FLOPs_recorded and rank==0:
                #     FLOPs_recorded = True
                #     flops = FlopCountAnalysis(model, (x, eps, init_residual, source, Sxf, Syf, self.source_mult))
                #     printc(f"flops per input device: {flops.total()/1e9/self.batch_size}G", 'r')
                #     with open(self.model_saving_path + '/'+'config.txt', 'a') as f:
                #         f.write(f'\nFLOPs per input device: {flops.total()/1e9/self.batch_size}(G)')
                
                if idx == 0 and rank==0:
                    xs = [init_x]
                    residuals = [init_residual]
                
                optimizer.zero_grad(set_to_none=True)
                r_new = init_residual.clone()
                
                # GMRES iterations:
                model.module.setup(eps, freq=dL/wl)

                bs = r_new.shape[0]
                
                V = []
                Z = []
                H = torch.zeros((bs, self.GMRES_iter + 1, self.GMRES_iter), dtype=torch.complex64).cuda()
                
                # with torch.no_grad():
                rhs = r2c(r_new)
                beta = (torch.sum(torch.conj(rhs)*rhs, dim=(1,2)) + 1e-6).sqrt()
                V.append((rhs/beta[:,None,None]))

                for j in range(self.GMRES_iter):
                    scale = torch.max(torch.mean(torch.abs(c2r(V[j])), dim=(1,2,3), keepdim=True), torch.tensor(1e-6))
                    z = r2c(model(c2r(V[j])/scale, freq=dL/wl)*scale)
                    Z.append(z)
                    w = r2c(self.Aop(c2r(z), eps[...,0], self.source_mult*source, (Sxf, Sxb), (Syf,Syb), dL, wl, self.bc_mult))

                    H_j_column = [] 
                    for i in range(j + 1):
                        # H[:, i, j] = torch.sum(torch.conj(w)*V[i], dim=(1,2))
                        # w = w - H[:, i, j, None, None] * V[i]
                        H_j_column.append(torch.sum(torch.conj(w)*V[i], dim=(1,2)))
                        w = w - H_j_column[-1][:,None,None] * V[i]
                    # H[:, j + 1, j, ] = complex_max(torch.sum(torch.conj(w)*w, dim=(1,2)), 1e-6).sqrt()
                    # V.append((w/H[:, j + 1, j, None, None]))

                    H_j_column.append(complex_max(torch.sum(torch.conj(w)*w, dim=(1,2)), 1e-6).sqrt())
                    V.append(w/H_j_column[-1][:,None,None])

                    # H should be a tensor of shape (bs, j+1, j)
                    # H_j_column is a list of tensors of shape (bs,) (length = j+2)
                    H_column = torch.stack(H_j_column, dim=1)[:,:,None] # shape (bs, j+2, 1)
                    if j == 0:
                        H = H_column # shape (bs, 2, 1)
                    else:
                        extra_row = torch.zeros((bs,1,j), device=eps.device)
                        H = torch.cat([H, extra_row], dim=1) # shape (bs, j+2, j)
                        H = torch.cat([H, H_column], dim=2) # shape (bs, j+2, j+1)

                e1 = torch.zeros((bs, self.GMRES_iter + 1), dtype=torch.complex64, device=eps.device)
                e1[:,0] = beta

                y, residual_norm, _, _ = torch.linalg.lstsq(H[:, :self.GMRES_iter + 1, :self.GMRES_iter], e1, rcond=None)

                x = init_x.clone()
                for i in range(self.GMRES_iter):
                    x = x + c2r(y[:, i, None, None] * Z[i])

                residual = c2r(rhs) - self.Aop(x, eps[...,0], self.source_mult*source, (Sxf, Sxb), (Syf,Syb), dL, wl, self.bc_mult)

                if idx == 0 and rank==0:
                    xs.append(x)
                    residuals.append(residual)

                loss = MAE_loss(residual) + MSE_loss(residual)

                loss.backward()
                optimizer.step()

                if rank==0:
                    with torch.no_grad():
                        ori_residual = self.residual_fn(torch.zeros_like(x), eps[...,0], top_bc, bottom_bc, left_bc, right_bc, self.source_mult*source, (Sxf, Sxb), (Syf,Syb), dL, wl, self.bc_mult, clamp=self.residual_clip)
                        rel_residual_loss = MAE_loss(residual)/ MAE_loss(ori_residual)

                    if step == start_epoch and idx == 0:
                        running_residual_loss = rel_residual_loss
                    else:
                        running_residual_loss = 0.95*running_residual_loss + 0.05*rel_residual_loss
                    pbar.set_description(f"running residual loss={running_residual_loss:.2e}")

                if (idx + 1) % 20 == 0 and rank==0:
                    printc(f'Epoch [{step + 1}/{self.epoch}], Step [{idx + 1}/{len(train_loader)}], residual loss: {running_residual_loss.item():.4f}', 'b')

                if gradient_count >= lr_update_steps:
                    gradient_count = 0
                    lr_scheduler.step()

                if rank==0 and idx == 0:
                    plot_data = [eps[0,:,:,0],self.source_mult*source[0,:,:,0]+source[0,:,:,1],Sxf[0,:,:]+Syf[0,:,:], init_residual[0,:,:,0]]
                    plot_title = ['eps', 'source', 'pml', 'residual_input']
                    cmaps = ['binary', 'seismic', None, 'seismic']
                    center_zero = [False, True, False, True]
                    row, column = 1, 4
                    plot_helper(plot_data,row,column,plot_title,self.model_saving_path+f"/plots/epoch_{step}_wl_{wl[0]:.2e}_dL_{dL[0]:.2e}_residual_debug.png", cmaps=cmaps, center_zero=center_zero)
                    
                    plot_data = [*[x[0,:,:,0] for x in xs], *[r[0,:,:,0] for r in residuals]]
                    plot_title = [*[f"output_{i}" for i in range(len(xs))], *[f"residual_{i}" for i in range(len(residuals))]]
                    cmaps = ['seismic']*len(xs) + ['seismic']*len(residuals)
                    center_zero = [True]*len(xs) + [True]*len(residuals)
                    row, column = 2, len(xs)
                    plot_helper(plot_data,row,column,plot_title,self.model_saving_path+f"/plots/epoch_{step}_wl_{wl[0]:.2e}_dL_{dL[0]:.2e}_debug_recurrent.png", cmaps=cmaps, center_zero=center_zero)

            #Save the weights at the end of each epoch
            checkpoint = {
                        'epoch': step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer,
                        'lr_scheduler': lr_scheduler
                    }
            torch.save(checkpoint, self.model_saving_path+"/last_model.pt")

            # evaluation

            # to save time, use running average to approximate loss:
            train_residual_loss = running_residual_loss

            test_residual_loss = 0
            if step % 2 == 0:
                model.eval()
                for idx, sample_batched in enumerate(test_loader):
                    eps, source, dL, wl = sample_batched['eps'].cuda(non_blocking=True), sample_batched['source'].cuda(non_blocking=True), sample_batched['dL'].cuda(non_blocking=True), sample_batched['wl'].cuda(non_blocking=True)
                    top_bc, bottom_bc, left_bc, right_bc = sample_batched['top_bc'].cuda(non_blocking=True), sample_batched['bottom_bc'].cuda(non_blocking=True), sample_batched['left_bc'].cuda(non_blocking=True), sample_batched['right_bc'].cuda(non_blocking=True)
                    Sxf, Sxb, Syf, Syb = sample_batched['sx_f'].cuda(non_blocking=True), sample_batched['sx_b'].cuda(non_blocking=True), sample_batched['sy_f'].cuda(non_blocking=True), sample_batched['sy_b'].cuda(non_blocking=True)

                    with torch.no_grad():
                        x = torch.zeros_like(source)
                        r_new = self.residual_fn(x, eps[...,0], top_bc, bottom_bc, left_bc, right_bc, self.source_mult*source, (Sxf, Sxb), (Syf,Syb), dL, wl, self.bc_mult, clamp=self.residual_clip)
                        if idx == 0 and rank==0:
                            xs = [x]
                            residuals = [r_new.clone()]

                        model.module.setup(eps, freq=dL/wl)
                        bs = r_new.shape[0]
                
                        V = []
                        Z = []
                        H = torch.zeros((bs, self.GMRES_iter + 1, self.GMRES_iter), dtype=torch.complex64).cuda()
                        
                        rhs = r2c(r_new)
                        beta = (torch.sum(torch.conj(rhs)*rhs, dim=(1,2)) + 1e-6).sqrt()
                        V.append((rhs/beta[:,None,None]))
                            
                        for j in range(self.GMRES_iter):
                            scale = torch.max(torch.mean(torch.abs(c2r(V[j])), dim=(1,2,3), keepdim=True), torch.tensor(1e-6))
                            z = r2c(model(c2r(V[j])/scale, freq=dL/wl)*scale)
                            Z.append(z)
                            w = r2c(self.Aop(c2r(z), eps[...,0], self.source_mult*source, (Sxf, Sxb), (Syf,Syb), dL, wl, self.bc_mult))
                            for i in range(j + 1):
                                H[:, i, j] = torch.sum(torch.conj(w)*V[i], dim=(1,2))
                                w = w - H[:, i, j, None, None] * V[i]
                            H[:, j + 1, j, ] = complex_max(torch.sum(torch.conj(w)*w, dim=(1,2)), 1e-6).sqrt()
                            V.append((w/H[:, j + 1, j, None, None]))

                        e1 = torch.zeros((bs, self.GMRES_iter + 1), dtype=torch.complex64).to(rhs.device)
                        e1[:,0] = beta

                        y, residual_norm, _, _ = torch.linalg.lstsq(H[:, :self.GMRES_iter + 1, :self.GMRES_iter], e1, rcond=None)

                        for i in range(self.GMRES_iter):
                            x = x + c2r(y[:, i, None, None] * Z[i])

                        residual = c2r(rhs) - self.Aop(x, eps[...,0], self.source_mult*source, (Sxf, Sxb), (Syf,Syb), dL, wl, self.bc_mult)

                        if idx == 0 and rank==0:
                            xs.append(x)
                            residuals.append(residual)

                        ori_residual = self.residual_fn(torch.zeros_like(x), eps[...,0], top_bc, bottom_bc, left_bc, right_bc, self.source_mult*source, (Sxf, Sxb), (Syf,Syb), dL, wl, self.bc_mult, clamp=self.residual_clip)
                        rel_residual_loss = MAE_loss(residual)/ MAE_loss(ori_residual)

                        test_residual_loss += rel_residual_loss.item()

                        if idx == 0 and rank==0:
                            plot_data = [eps[0,:,:,0],self.source_mult*source[0,:,:,0]+source[0,:,:,1],Sxf[0,:,:]+Syf[0,:,:],\
                                         xs[-1][0,:,:,0], residuals[-1][0,:,:,0]]
                            plot_title = ['eps', 'source', 'pml', 'output', 'residual']
                            cmaps = ['binary', 'seismic', None, 'seismic', 'seismic']
                            center_zero = [False, True, False, True, True]
                            row, column = 1, 5
                            plot_helper(plot_data,row,column,plot_title,self.model_saving_path+f"/plots/epoch_{step}_wl_{wl[0]:.2e}_dL_{dL[0]:.2e}.png", cmaps=cmaps, center_zero=center_zero)

                            plot_data = [*[x[0,:,:,0] for x in xs], *[r[0,:,:,0] for r in residuals]]
                            plot_title = [*[f"output_{i}" for i in range(len(xs))], *[f"residual_{i}" for i in range(len(residuals))]]
                            cmaps = ['seismic']*len(xs) + ['seismic']*len(residuals)
                            center_zero = [True]*len(xs) + [True]*len(residuals)
                            row, column = 2, len(xs)
                            plot_helper(plot_data,row,column,plot_title,self.model_saving_path+f"/plots/epoch_{step}_wl_{wl[0]:.2e}_dL_{dL[0]:.2e}_recurrent.png", cmaps=cmaps, center_zero=center_zero)
    
                test_residual_loss /= len(test_loader)

                if rank==0:
                    printc(f"train_residual_loss: {train_residual_loss:.2e}, test_residual_loss: {test_residual_loss:.2e}", 'r')
                    
                    new_df = pd.DataFrame([[step+1,str(lr_scheduler.get_last_lr()),train_residual_loss, test_residual_loss]], \
                                        columns=['epoch', 'lr', 'train_residual_loss', 'test_residual_loss'])
                    df = pd.concat([df,new_df])

                    df.to_csv(self.model_saving_path + '/'+'df.csv',index=False)

                    if(test_residual_loss<best_loss):
                        best_loss = test_residual_loss
                        checkpoint = {
                                        'epoch': step,
                                        'state_dict': model.state_dict(),
                                        'optimizer': optimizer,
                                        'lr_scheduler': lr_scheduler
                                    }
                        torch.save(checkpoint, self.model_saving_path+"/best_model.pt")
            if rank==0:
                epoch_stop_time = timeit.default_timer()
                printc(f"epoch run time: {epoch_stop_time-epoch_start_time:.2f}s", 'r')
