# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np

from lpips import LPIPS
from pytorch_msssim import SSIM
from fused_ssim import fused_ssim


metric_networks = {}


def l1_loss(x, y):
    return torch.nn.functional.l1_loss(x, y)

def l2_loss(x, y):
    return torch.nn.functional.mse_loss(x, y)

def huber_loss(x, y, thres=0.01):
    l1 = (x - y).abs().mean(0)
    l2 = (x - y).pow(2).mean(0)
    loss = torch.where(
        l1 < thres,
        l2,
        2 * thres * l1 - thres ** 2)
    return loss.mean()

def psnr_score(x, y):
    return -10 * torch.log10(l2_loss(x, y))

def ssim_score(x, y):
    if 'SSIM' not in metric_networks:
        metric_networks['SSIM'] = SSIM(data_range=1, win_size=11, win_sigma=1.5, channel=3).cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks['SSIM'](x, y)

def ssim_loss(x, y):
    return 1 - ssim_score(x, y)

def fast_ssim_loss(x, y):
    # Note! Only x get gradient in backward.
    is_train = x.requires_grad or y.requires_grad
    return 1 - fused_ssim(x.unsqueeze(0), y.unsqueeze(0), padding="valid", train=is_train)

def lpips_loss(x, y, net='vgg'):
    key = f'LPIPS_{net}'
    if key not in metric_networks:
        metric_networks[key] = LPIPS(net=net, version='0.1').cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks[key](x, y)

def correct_lpips_loss(x, y, net='vgg'):
    key = f'LPIPS_{net}'
    if key not in metric_networks:
        metric_networks[key] = LPIPS(net=net, version='0.1').cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks[key](x*2-1, y*2-1)

def entropy_loss(prob):
    pos_prob = prob.clamp(1e-6, 1-1e-6)
    neg_prob = 1 - pos_prob
    return -(pos_prob * pos_prob.log() + neg_prob * neg_prob.log()).mean()

def prob_concen_loss(prob):
    return (prob.square() * (1 - prob).square()).mean()


def exp_anneal(end_mul, iter_now, iter_from, iter_end):
    if end_mul == 1 or iter_now >= iter_end:
        return 1
    total_len = iter_end - iter_from + 1
    now_len = max(0, iter_now - iter_from + 1)
    now_p = min(1.0, now_len / total_len)
    return end_mul ** now_p


class NormalDepthConsistencyLoss:
    def __init__(self, iter_from, iter_end, ks, tol_deg):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.ks = ks
        self.tol_cos = np.cos(np.deg2rad(tol_deg))

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_T" in render_pkg, "Forgot to set `output_T=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        # Read rendering results
        render_alpha = 1 - render_pkg['raw_T'].detach().squeeze(0)
        render_depth = render_pkg['raw_depth'][0]
        render_normal = render_pkg['raw_normal']

        # Compute depth to normal
        N_mean = cam.depth2normal(render_depth, ks=self.ks, tol_cos=self.tol_cos)

        # Blend with alpha and compute target
        target = render_alpha.square()
        N_mean = N_mean * render_alpha

        # Compute loss
        mask = (N_mean != 0).any(0)
        loss_map = (target - (render_normal * N_mean).sum(dim=0)) * mask
        loss = loss_map.mean()
        return loss


class NormalMedianConsistencyLoss:
    def __init__(self, iter_from, iter_end):
        self.iter_from = iter_from
        self.iter_end = iter_end

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        # TODO: median depth is not differentiable
        render_median = render_pkg['raw_depth'][2]
        render_normal = render_pkg['raw_normal']

        # Compute depth to normal
        N_med = cam.depth2normal(render_median, ks=3)

        # Compute loss
        mask = (N_med != 0).any(0)
        loss_map = (1 - (render_normal * N_med).sum(dim=0)) * mask
        loss = loss_map.mean()
        return loss
