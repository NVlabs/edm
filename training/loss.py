# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from typing import Callable

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

@persistence.persistent_class
class MonotonicEDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.lambda_mean = (-2) * self.P_mean
        self.lambda_std = 2 * self.P_std
        self.sigma_data = sigma_data
        self.argmax_lambda = torch.tensor(-3.30778)
        self.weight_lambda_max = ((torch.exp(-self.argmax_lambda) + self.sigma_data ** 2) /
                                  (torch.exp(-self.argmax_lambda) * self.sigma_data ** 2))
        
    def normal_pdf(self, x, mu=0, sigma=1):
        return (1 / (sigma * (2 * torch.pi) ** 0.5)) * torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # argmax_lambda w(lambda) = -3.30778
    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        lambda_ = (-1) * (sigma ** 2).log()
        weight = torch.where(
            lambda_ < self.argmax_lambda,
            (self.weight_lambda_max * torch.exp(-self.argmax_lambda) * 
            self.normal_pdf(self.argmax_lambda, self.lambda_mean, self.lambda_std) / 
            (torch.exp(-lambda_) * self.normal_pdf(lambda_, self.lambda_mean, self.lambda_std)) 
            ),
            (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        )
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

@persistence.persistent_class
class ULoss:
    def __call__(self, net, images, labels=None, augment_pipe=None):
        t = torch.rand([images.shape[0]], device=images.device) * (net.module.t_max - net.module.t_min) + net.module.t_min
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        eps = torch.randn_like(y)
        D_yn = net(y * net.module.alpha(t).to(torch.float32).reshape(-1, 1, 1, 1) + 
                   eps * net.module.sigma(t).to(torch.float32).reshape(-1, 1, 1, 1), 
                   t, labels, augment_labels=augment_labels)
        loss = (D_yn - eps * net.module.u(t).to(torch.float32).reshape(-1, 1, 1, 1)) ** 2
        return loss

#----------------------------------------------------------------------------