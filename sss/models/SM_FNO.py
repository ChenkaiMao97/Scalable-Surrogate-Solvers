# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

import gin

################################################################
# fourier layer
################################################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, ALPHA, stride=1):
        super(BasicBlock, self).__init__()
        self.ALPHA = ALPHA
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(nn.Identity())
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=self.ALPHA)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.leaky_relu(out, negative_slope=self.ALPHA)
        return out

class BasicBlock_without_shortcut(nn.Module):
    def __init__(self, in_planes, planes, ALPHA, stride=1):
        super(BasicBlock_without_shortcut, self).__init__()
        self.ALPHA = ALPHA
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=self.ALPHA)
        out = self.conv2(out)
        return out


class Modulated_SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_freq, modes1, modes2):
        super(Modulated_SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_freq = hidden_freq
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (self.in_channels*self.out_channels))

        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 2, self.hidden_freq))
        self.mlp = nn.Linear(self.hidden_freq, self.modes1*self.modes2*2, bias=False)
        
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,bioxy->boxy", input, weights)

    def forward(self, x, mod1, mod2):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        weights = self.mlp(self.weights).reshape((self.in_channels, self.out_channels, 2, self.modes1, self.modes2, 2))

        complex_mult1 = torch.view_as_complex(weights[:,:,0,:,:,:])*mod1
        complex_mult2 = torch.view_as_complex(weights[:,:,1,:,:,:])*mod2
        
        # for DDP:
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], complex_mult1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], complex_mult2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


@gin.configurable
class FNO_multimodal_2d(nn.Module):
    def __init__(
        self, 
        domain_sizex, 
        domain_sizey,
        f_padding,
        f_modes, 
        HIDDEN_DIM, 
        hidden_freq, 
        mod_data_channels,
        pre_data_channels,
        outc,
        num_fourier_layers,
        ALPHA,
    ):
        super(FNO_multimodal_2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = f_modes
        self.modes2 = f_modes
        self.width = HIDDEN_DIM
        self.hidden_freq = hidden_freq
        self.padding = f_padding # pad the domain if input is non-periodic
        self.sizex = domain_sizex
        self.sizey = domain_sizey

        self.mod_data_channels = mod_data_channels
        self.pre_data_channels = pre_data_channels
        self.fc0_dielectric = nn.Linear(self.pre_data_channels, self.width) # input channel is 3: (a(x, y), x, y), 

        self.num_fourier_layers = num_fourier_layers
        self.ALPHA = ALPHA

        self.convs = []
        self.ws = []
        for i in range(self.num_fourier_layers):
            self.convs.append(Modulated_SpectralConv2d(self.width, self.width, self.hidden_freq, self.modes1, self.modes2))
            self.ws.append(BasicBlock_without_shortcut(self.width, self.width, self.ALPHA))
        self.convs = nn.ModuleList(self.convs)
        self.ws = nn.ModuleList(self.ws)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, outc)

        # the modulation branch of the network:
        self.m_basic1 = BasicBlock(self.mod_data_channels, self.width, self.ALPHA, 1)
        self.m_basic2 = BasicBlock(self.width, self.width, self.ALPHA, 1)
        self.m_basic3 = BasicBlock(self.width, self.width, self.ALPHA, 1)
        self.m_bc1 = nn.Linear(int(self.width*self.sizex*self.sizey/64), self.modes1*self.modes2)

        self.m_bc2_1 = nn.Linear(self.modes1*self.modes2, self.modes1*self.modes2)
        self.m_bc2_2 = nn.Linear(self.modes1*self.modes2, self.modes1*self.modes2)

    def forward(self, x_current, eps, residue, source, Sx_imag, Sy_imag, source_mult):
        # Sx_f: [bs, subdomain_size, subdomain_size,2]
        # Sy_f: [bs, subdomain_size, subdomain_size,2]
        # source: [bs, subdomain_size, subdomain_size, 2]
        # eps: [bs, subdomain_size, subdomain_size]
        # top_bc, bottom_bc: [bs, 1, subdomain_size, 2]
        # left_bc, right_bc: [bs, subdomain_size, 1, 2]

        batch_size = eps.shape[0]

        grid = self.get_grid(eps.shape, eps.device)
        
        pre_data = torch.cat((x_current.permute((0,3,1,2)), eps.unsqueeze(dim=1), residue.permute((0,3,1,2)), source_mult*source.permute((0,3,1,2)), Sx_imag.unsqueeze(dim=1), Sy_imag.unsqueeze(dim=1), grid), dim=1)
        mod_data = pre_data
        # modulating branch:
        mod = F.avg_pool2d(self.m_basic1(mod_data),2)
        mod = F.avg_pool2d(self.m_basic2(mod),2)
        mod = F.avg_pool2d(self.m_basic3(mod),2)
        mod = self.m_bc1(mod.reshape((batch_size, -1)))
        mod1 = self.m_bc2_1(mod).reshape((batch_size, 1,1, self.modes1,self.modes2))
        mod2 = self.m_bc2_2(mod).reshape((batch_size, 1,1, self.modes1,self.modes2))

        x = pre_data.permute(0, 2, 3, 1)
        x = self.fc0_dielectric(x)
        x = x.permute(0, 3, 1, 2)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.num_fourier_layers-1):
            x1 = self.convs[i](x, mod1, mod2)
            x2 = self.ws[i](x)
            x = x1 + x2 + x
            x = F.leaky_relu(x, negative_slope=self.ALPHA)

        x1 = self.convs[-1](x, mod1, mod2)
        x2 = self.ws[-1](x)
        x = x1 + x2 + x

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=self.ALPHA)
        x = self.fc2(x)
        # x = x.permute(0, 3, 1, 2)
        
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
