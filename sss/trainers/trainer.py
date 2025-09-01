# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import  os, sys, timeit
import  numpy as np
import pandas as pd

from sss.trainers import BaseTrainer
from sss.utils.PDE_utils import maxwell_robin_residue
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

def regConstScheduler(epoch, phys_start_epoch, ratio, last_epoch_data_loss, last_epoch_physical_loss):
    if(epoch<phys_start_epoch):
        return 0
    else:
        return ratio*last_epoch_data_loss/last_epoch_physical_loss


@gin.configurable
class Trainer(BaseTrainer):
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
        fix_point_iter: int = 1,
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
        self.fix_point_iter = fix_point_iter
        printc(f"training with fix_point_iter: {self.fix_point_iter}", 'g')
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
                    df = pd.DataFrame(columns=['epoch', 'lr', 'train_data_loss', 'train_residue_loss', 'test_data_loss', 'test_residue_loss'])
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
                df = pd.DataFrame(columns=['epoch', 'lr', 'train_data_loss', 'train_residue_loss', 'test_data_loss', 'test_residue_loss'])
            
            model = self.model_fn(domain_sizex = self.domain_sizex, domain_sizey = self.domain_sizey, f_padding = self.f_padding)
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
            sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=each_GPU_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=test_sampler)

        FLOPs_recorded = False
        gradient_count = 0
        best_loss = 1e4
        running_data_loss = 1.0
        running_residue_loss = 1.0
        last_epoch_data_loss = df.iloc[-1]['test_data_loss'] if self.continue_train else 1.
        last_epoch_physical_loss = df.iloc[-1]['test_residue_loss'] if self.continue_train else 1.

        for step in range(start_epoch, self.epoch):
            train_sampler.set_epoch(step)
            test_sampler.set_epoch(step)
            epoch_start_time = timeit.default_timer()
            reg_norm = regConstScheduler(step, self.phys_start_epoch, self.ratio, last_epoch_data_loss, last_epoch_physical_loss)
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

                y_batch_train, source_batch_train, Sxf_batch_train, Syf_batch_train, Sxb_batch_train, Syb_batch_train, eps_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train, dL_train, wl_train = sample_batched['field'].cuda(non_blocking=True), sample_batched['source'].cuda(non_blocking=True), sample_batched['Sx_f'].cuda(non_blocking=True), sample_batched['Sy_f'].cuda(non_blocking=True), sample_batched['Sx_b'].cuda(non_blocking=True), sample_batched['Sy_b'].cuda(non_blocking=True), sample_batched['eps'].cuda(non_blocking=True), sample_batched['top_bc'].cuda(non_blocking=True), sample_batched['bottom_bc'].cuda(non_blocking=True), sample_batched['left_bc'].cuda(non_blocking=True), sample_batched['right_bc'].cuda(non_blocking=True), sample_batched['dL'].cuda(non_blocking=True), sample_batched['wl'].cuda(non_blocking=True)

                x = 0.01*torch.randn_like(y_batch_train, dtype=torch.float32)
                init_residue = self.residual_fn(x, eps_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train, source_batch_train, (Sxf_batch_train, Sxb_batch_train), (Syf_batch_train,Syb_batch_train), dL_train, wl_train, self.bc_mult, clamp=self.residual_clip)

                # record FLOPs for once:
                if not FLOPs_recorded and rank==0:
                    FLOPs_recorded = True
                    flops = FlopCountAnalysis(model, (x, eps_batch_train, init_residue, source_batch_train, Sxf_batch_train, Syf_batch_train, self.source_mult))
                    printc(f"flops per input device: {flops.total()/1e9/self.batch_size}G", 'r')
                    with open(self.model_saving_path + '/'+'config.txt', 'a') as f:
                        f.write(f'\nFLOPs per input device: {flops.total()/1e9/self.batch_size}(G)')
                
                if idx == 0 and rank==0:
                    xs = []
                    residuals = [init_residue]
                
                optimizer.zero_grad(set_to_none=True)
                r_new = init_residue.clone()
                for iter in range(self.fix_point_iter):
                    residue_scale = torch.maximum(torch.mean(torch.abs(r_new), dim=(1,2,3), keepdim=True), torch.tensor(1e-3))
                    x_scale = torch.maximum(torch.mean(torch.abs(x), dim=(1,2,3), keepdim=True), torch.tensor(1e-3))
                    error = residue_scale*model(x/x_scale, eps_batch_train, r_new/residue_scale, source_batch_train, Sxf_batch_train, Syf_batch_train, self.source_mult)
                    x = x - error
                    r_new = self.residual_fn(x, eps_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train, source_batch_train, (Sxf_batch_train, Sxb_batch_train), (Syf_batch_train,Syb_batch_train), dL_train, wl_train, self.bc_mult, clamp=self.residual_clip)

                    if idx == 0 and rank==0:
                        xs.append(x)
                        residuals.append(r_new)

                    residual_loss = MAE_loss(r_new)
                    data_loss = MAE_loss(x, y_batch_train)

                    loss = data_loss + reg_norm*residual_loss 

                    loss.backward()
                    x = x.detach()
                    r_new = r_new.detach()

                optimizer.step()

                if step == start_epoch and idx == 0:
                    running_data_loss = data_loss.item()
                    running_residue_loss = residual_loss.item()
                else:
                    running_data_loss = 0.95*running_data_loss + 0.05*data_loss.item()
                    running_residue_loss = 0.95*running_residue_loss + 0.05*residual_loss.item()
                if rank==0:
                    pbar.set_description(f"running data loss={running_data_loss:.2e}, running residue loss={running_residue_loss:.2e}")

                if (idx + 1) % 20 == 0 and rank==0:
                    printc(f'Epoch [{step + 1}/{self.epoch}], Step [{idx + 1}/{len(train_loader)}], data loss: {running_data_loss.item():.4f}, residue loss: {running_residue_loss.item():.4f}', 'b')

                if gradient_count >= lr_update_steps:
                    gradient_count = 0
                    lr_scheduler.step()

                if rank==0 and idx == 0:
                    residue_gt = self.residual_fn(y_batch_train[:1], eps_batch_train[:1], top_bc_train[:1], bottom_bc_train[:1], left_bc_train[:1], right_bc_train[:1], source_batch_train[:1], (Sxf_batch_train[:1], Sxb_batch_train[:1]), (Syf_batch_train[:1],Syb_batch_train[:1]), dL_train[:1], wl_train[:1], self.bc_mult, clamp=self.residual_clip)

                    plot_data = [eps_batch_train[0,:,:],self.source_mult*source_batch_train[0,:,:,0]+source_batch_train[0,:,:,1],Sxf_batch_train[0,:,:]+Syf_batch_train[0,:,:],y_batch_train[0,:,:,0], residue_gt[0,:,:,0], init_residue[0,:,:,0]]
                    plot_title = ['eps', 'source', 'pml', 'gt', 'residue_gt', 'residue_input']
                    row, column = 1, 6
                    plot_helper(plot_data,row,column,plot_title,self.model_saving_path+"/plots/epoch_"+str(step)+"_residue_debug.png")
                    
                    plot_data = [y_batch_train[0,:,:,0], *[y_batch_train[0,:,:,0]-x[0,:,:,0] for x in xs], *[r[0,:,:,0] for r in residuals]]
                    plot_title = ["gt", *[f"error_{i}" for i in range(1,self.fix_point_iter+1)], "init_residual", *[f"residual_{i}" for i in range(1,self.fix_point_iter+1)]]
                    row, column = 2, self.fix_point_iter+1
                    plot_helper(plot_data,row,column,plot_title,self.model_saving_path+"/plots/epoch_"+str(step)+"_debug_recurrent.png")

            #Save the weights at the end of each epoch
            checkpoint = {
                        'epoch': step,
                        'model': model,
                        'optimizer': optimizer,
                        'lr_scheduler': lr_scheduler
                    }
            torch.save(checkpoint, self.model_saving_path+"/last_model.pt")

            # evaluation

            # to save time, use running average to approximate loss:
            train_data_loss = running_data_loss
            train_residue_loss = running_residue_loss

            test_data_loss = 0
            test_residue_loss = 0
            if step % 2 == 0:
                model.eval()
                for idx, sample_batched in enumerate(test_loader):
                    y_batch_test, source_batch_test, Sxf_batch_test, Syf_batch_test, Sxb_batch_test, Syb_batch_test, eps_batch_test, top_bc_test, bottom_bc_test, left_bc_test, right_bc_test, dL_test, wl_test = sample_batched['field'].cuda(), sample_batched['source'].cuda(), sample_batched['Sx_f'].cuda(), sample_batched['Sy_f'].cuda(), sample_batched['Sx_b'].cuda(), sample_batched['Sy_b'].cuda(), sample_batched['eps'].cuda(), sample_batched['top_bc'].cuda(), sample_batched['bottom_bc'].cuda(), sample_batched['left_bc'].cuda(), sample_batched['right_bc'].cuda(), sample_batched['dL'].cuda(), sample_batched['wl'].cuda()

                    with torch.no_grad():
                        x = torch.randn_like(y_batch_test)*0.01
                        r_new = self.residual_fn(x, eps_batch_test, top_bc_test, bottom_bc_test, left_bc_test, right_bc_test, source_batch_test, (Sxf_batch_test, Sxb_batch_test), (Syf_batch_test,Syb_batch_test), dL_test, wl_test, self.bc_mult, clamp=self.residual_clip)
                        if idx == 0 and rank==0:
                            xs = []
                            residuals = [r_new.clone()]
                        for iter in range(self.fix_point_iter):
                            residue_scale = torch.maximum(torch.mean(torch.abs(r_new), dim=(1,2,3), keepdim=True), torch.tensor(1e-3))
                            x_scale = torch.maximum(torch.mean(torch.abs(x), dim=(1,2,3), keepdim=True), torch.tensor(1e-3))
                            error = residue_scale*model(x/x_scale, eps_batch_test, r_new/residue_scale, source_batch_test, Sxf_batch_test, Syf_batch_test, self.source_mult)
                            x = x - error
                            r_new = self.residual_fn(x, eps_batch_test, top_bc_test, bottom_bc_test, left_bc_test, right_bc_test, source_batch_test, (Sxf_batch_test, Sxb_batch_test), (Syf_batch_test,Syb_batch_test), dL_test, wl_test, self.bc_mult, clamp=self.residual_clip)

                            if idx == 0 and rank==0:
                                xs.append(x)
                                residuals.append(r_new)
                        
                        data_loss = MAE_loss(x, y_batch_test)
                        residual_loss = MAE_loss(r_new)

                        test_data_loss += data_loss.item()
                        test_residue_loss += residual_loss.item()

                        if idx == 0 and rank==0:
                            plot_data = [eps_batch_test[0,:,:],self.source_mult*source_batch_test[0,:,:,0]+source_batch_test[0,:,:,1],Sxf_batch_test[0,:,:]+Syf_batch_test[0,:,:],\
                                        y_batch_test[0,:,:,0], y_batch_test[0,:,:,0]-xs[-1][0,:,:,0], torch.abs(residuals[-1][0,:,:,0]+1j*residuals[-1][0,:,:,1])]
                            plot_title = ['eps', 'source', 'pml', 'gt_r', 'error_r', 'residual_abs']
                            row, column = 2, 3
                            plot_helper(plot_data,row,column,plot_title,self.model_saving_path+"/plots/epoch_"+str(step)+".png")

                            plot_data = [y_batch_test[0,:,:,0], *[y_batch_test[0,:,:,0]-x[0,:,:,0] for x in xs], *[r[0,:,:,0] for r in residuals]]
                            plot_title = ["gt", *[f"error_{i}" for i in range(1,self.fix_point_iter+1)], "init_residual", *[f"residual_{i}" for i in range(1,self.fix_point_iter+1)]]
                            row, column = 2, self.fix_point_iter+1
                            plot_helper(plot_data,row,column,plot_title,self.model_saving_path+"/plots/epoch_"+str(step)+"_recurrent.png")
    
                test_data_loss /= len(test_loader)
                test_residue_loss /= len(test_loader)
                last_epoch_data_loss = test_data_loss
                last_epoch_physical_loss = test_residue_loss
        

                if rank==0:
                    print('train loss: %.5f, test loss: %.5f' % (train_data_loss, test_data_loss), flush=True)
                    new_df = pd.DataFrame([[step+1,str(lr_scheduler.get_last_lr()),train_data_loss, train_residue_loss, test_data_loss, test_residue_loss]], \
                                        columns=['epoch', 'lr', 'train_data_loss', 'train_residue_loss', 'test_data_loss', 'test_residue_loss'])
                    df = pd.concat([df,new_df])

                    df.to_csv(self.model_saving_path + '/'+'df.csv',index=False)

                    if(test_data_loss<best_loss):
                        best_loss = test_data_loss
                        checkpoint = {
                                        'epoch': step,
                                        'model': model,
                                        'optimizer': optimizer,
                                        'lr_scheduler': lr_scheduler
                                    }
                        torch.save(checkpoint, self.model_saving_path+"/best_model.pt")
            if rank==0:
                epoch_stop_time = timeit.default_timer()
                printc(f"epoch run time: {epoch_stop_time-epoch_start_time:.2f}s", 'r')
