# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import torch
from svraster_cuda.meta import MAX_NUM_LEVELS

from src.utils import octree_utils

class SVInOut:

    def save(self, path):
        '''
        Save the necessary attributes and parameters for reproducing rendering.
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            'active_sh_degree': self.active_sh_degree,
            'ss': self.ss,
            'scene_center': self.scene_center.cpu(),
            'inside_extent': self.inside_extent.cpu(),
            'scene_extent': self.scene_extent.cpu(),
            'octpath': self.octpath.cpu(),
            'octlevel': self.octlevel.cpu(),
            '_geo_grid_pts': self._geo_grid_pts.detach().cpu(),
            '_sh0': self._sh0.detach().cpu(),
            '_shs': self._shs.detach().cpu(),
        }
        torch.save(state_dict, path)
        self.latest_save_path = path

    def load(self, path):
        '''
        Load the saved models.
        '''
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        self.active_sh_degree = state_dict['active_sh_degree']
        self.ss = state_dict['ss']

        self.scene_center = state_dict['scene_center'].cuda()
        self.inside_extent = state_dict['inside_extent'].cuda()
        self.scene_extent = state_dict['scene_extent'].cuda()

        self.octpath = state_dict['octpath'].cuda()
        self.octlevel = state_dict['octlevel'].cuda().to(torch.int8)
        self.vox_center, self.vox_size = octree_utils.octpath_decoding(
            self.octpath, self.octlevel, self.scene_center, self.scene_extent)
        self.grid_pts_key, self.vox_key = octree_utils.build_grid_pts_link(self.octpath, self.octlevel)

        self._geo_grid_pts = state_dict['_geo_grid_pts'].cuda().requires_grad_()
        self._sh0 = state_dict['_sh0'].cuda().requires_grad_()
        self._shs = state_dict['_shs'].cuda().requires_grad_()

        N = len(self.octpath)
        self._subdiv_p = torch.full([N, 1], 1.0, dtype=torch.float32, device="cuda").requires_grad_()
        self.subdiv_meta = torch.zeros([N, 1], dtype=torch.float32, device="cuda")

        self.bg_color = torch.tensor(
            [1, 1, 1] if self.white_background else [0, 0, 0],
            dtype=torch.float32, device="cuda")

        self.loaded_path = path

    def save_iteration(self, iteration):
        path = os.path.join(self.model_path, "checkpoints", f"iter{iteration:06d}_model.pt")
        self.save(path)
        self.latest_save_iter = iteration

    def load_iteration(self, iteration=-1):
        if iteration == -1:
            # Find the maximum iteration if it is -1.
            fnames = os.listdir(os.path.join(self.model_path, "checkpoints"))
            loaded_iter = max(int(re.sub("[^0-9]", "", fname)) for fname in fnames)
        else:
            loaded_iter = iteration

        path = os.path.join(self.model_path, "checkpoints", f"iter{loaded_iter:06d}_model.pt")
        self.load(path)

        self.loaded_iter = iteration

        return loaded_iter
