# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from src.utils import activation_utils
from src.utils import octree_utils

'''
Adaptive sparse voxel pruning and subdivision.
There are three types of data mode to tackle.

1. Per-voxel attribute:
   Each voxel has it's own non-trainable data field.

2. Per-voxel parameters:
   Similar to per-voxel attribute but we have to process the Adam optimizer state as well.

3. Grid points parameters:
   The trainable parameters are attached to the eight grid points of each voxel.
   A grid point parameter can be shared by adjacent voxels.
'''

class SVAdaptive:

    @torch.no_grad()
    def pruning(self, prune_mask):
        '''
        Prune sparse voxels.
        The grid points and the optimizer are updated accordingly.
        Input:
            @prune_mask      [N] Mask indicating the voxels to prune.
        '''
        kept_idx = (~prune_mask).argwhere().squeeze(1)
        self._prune_attr(kept_idx)
        self._prune_voxel_parameters(kept_idx)
        self._prune_grid_pts_parameters(kept_idx)

    @torch.no_grad()
    def subdividing(self, subdivide_mask, save_gpu=False):
        '''
        Prune sparse voxels.
        The grid points and the optimizer are updated accordingly.
        Input:
            @subdivide_mask  [N] Mask indicating the voxels to subdivide.
        '''
        # Compute voxel index to keep and to subdivided
        if len(subdivide_mask.shape) == 2:
            subdivide_mask = subdivide_mask.squeeze(1)
        kept_idx = (~subdivide_mask).argwhere().squeeze(1)
        subdivide_idx = subdivide_mask.argwhere().squeeze(1)

        # Subdivided the selected voxels into their eight octants
        self._subdivide_attr(kept_idx, subdivide_idx)
        self._subdivide_voxel_parameters(kept_idx, subdivide_idx)
        self._subdivide_grid_pts_parameters(kept_idx, subdivide_idx, save_gpu=save_gpu)

    @torch.no_grad()
    def sh_degree_add1(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @torch.no_grad()
    def compute_training_stat(self, camera_lst):
        '''
        Compute the following statistic of each voxel from the given cameras.
        1. max_w:             the maximum blending weight.
        2. min_samp_interval: the minimum sampling interval (inverse of maximum sampling rate).
        3. view_cnt:          number of cameras with non-zero blending weight.

        Input:
            @camera_lst    [Camera, ...] A list of cameras.
        '''
        self.freeze_vox_geo()
        max_w = torch.zeros([self.num_voxels, 1], dtype=torch.float32, device="cuda")
        min_samp_interval = torch.full([self.num_voxels, 1], 1e30, dtype=torch.float32, device="cuda")
        view_cnt = torch.zeros([self.num_voxels, 1], dtype=torch.float32, device="cuda")
        for camera in camera_lst:
            max_w_i = self.render(camera, color_mode='dontcare', track_max_w=True)['max_w']
            max_w = torch.maximum(max_w, max_w_i)

            vis_idx = (max_w_i > 0).squeeze().argwhere().squeeze()
            zdist = ((self.vox_center[vis_idx] - camera.position) * camera.lookat).sum(-1, keepdims=True)
            samp_interval = zdist * camera.pix_size
            min_samp_interval[vis_idx] = torch.minimum(min_samp_interval[vis_idx], samp_interval)

            view_cnt[vis_idx] += 1

        stat_pkg = {
            'max_w': max_w,
            'min_samp_interval': min_samp_interval,
            'view_cnt': view_cnt,
        }
        self.unfreeze_vox_geo()
        return stat_pkg

    #################################################
    # The following are the low-level functions for
    # sparse voxels pruning and subdivision.
    #################################################
    @torch.no_grad()
    def _prune_attr(self, kept_idx):
        '''
        Prune non-trainable per-voxel attributes.
        Input:
            @kept_idx: the voxels to be kept.
        '''
        for name in self.per_voxel_attr_lst:
            ori_attr = getattr(self, name)
            new_attr = mask_cat_perm(ori_attr, kept_idx=kept_idx)
            setattr(self, name, new_attr)
            del ori_attr
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _subdivide_attr(self, kept_idx, subdivide_idx):
        '''
        Subdivide non-trainable per-voxel attributes.
        Input:
            @kept_idx: the voxels to be kept.
            @subdivide_idx: the voxels to be subdivided.
        '''
        octpath, octlevel = octree_utils.gen_children(
            self.octpath[subdivide_idx],
            self.octlevel[subdivide_idx])
        vox_center, vox_size = octree_utils.octpath_decoding(
            octpath,
            octlevel,
            self.scene_center,
            self.scene_extent)

        special_subdiv = dict(
            octpath=octpath,
            octlevel=octlevel,
            vox_center=vox_center,
            vox_size=vox_size,
        )

        for name in self.per_voxel_attr_lst:
            ori_attr = getattr(self, name)
            if name in special_subdiv:
                subdiv_attr = special_subdiv.pop(name)
            else:
                subdiv_attr = ori_attr[subdivide_idx].repeat_interleave(8, dim=0)
            new_attr = mask_cat_perm(
                ori_attr,
                kept_idx=kept_idx,
                cat_tensor=subdiv_attr)
            setattr(self, name, new_attr)
            del ori_attr, subdiv_attr
            torch.cuda.empty_cache()

        assert len(special_subdiv) == 0

    @torch.no_grad()
    def _prune_voxel_parameters(self, kept_idx):
        '''
        Prune trainable per-voxel parameters.
        Input:
            @kept_idx: the voxels to be kept.
        '''
        # Shortcut to access and manipulate optimizer
        if hasattr(self, 'optimizer'):
            state = self.optimizer.state
            param_groups = self.optimizer.param_groups
            lookup = {
                group['name']: idx
                for idx, group in enumerate(param_groups)
            }
        else:
            state = None

        # Update voxel trainable parameters
        for name in self.per_voxel_param_lst:
            param = getattr(self, name)
            if state is not None:
                state_dict = state.pop(param)
            ori_param = param.detach()
            del param
            torch.cuda.empty_cache()

            # Update parameter
            new_param = mask_cat_perm(
                ori_param,
                kept_idx=kept_idx).requires_grad_()
            setattr(self, name, new_param)
            del ori_param
            torch.cuda.empty_cache()

            # Update optimizer
            if state is not None:
                # WARNING: Cause error if a param is never optimized.
                assert len(param_groups[lookup[name]]['params']) == 1
                param_groups[lookup[name]]['params'][0] = new_param

                for k in self.state_attr_names:
                    ori_state = state_dict[k]
                    state_dict[k] = mask_cat_perm(
                        ori_state,
                        kept_idx=kept_idx)
                    assert state_dict[k].shape == new_param.shape
                    del ori_state
                    torch.cuda.empty_cache()
                state[new_param] = state_dict

            del new_param
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _subdivide_voxel_parameters(self, kept_idx, subdivide_idx):
        '''
        Subdivide trainable per-voxel parameters.
        Input:
            @kept_idx: the voxels to be kept.
            @subdivide_idx: the voxels to be subdivided.
        '''
        # Shortcut to access and manipulate optimizer
        state = self.optimizer.state
        param_groups = self.optimizer.param_groups
        lookup = {
            group['name']: idx
            for idx, group in enumerate(param_groups)
        }

        # Update voxel trainable parameters
        for name in self.per_voxel_param_lst:
            param = getattr(self, name)
            state_dict = state.pop(param)
            ori_param = param.detach()
            del param
            torch.cuda.empty_cache()

            # Update parameter
            subdiv_param = ori_param[subdivide_idx].repeat_interleave(8, dim=0)
            new_param = mask_cat_perm(
                ori_param,
                kept_idx=kept_idx,
                cat_tensor=subdiv_param).requires_grad_()
            setattr(self, name, new_param)
            del ori_param, subdiv_param
            torch.cuda.empty_cache()

            # Update optimizer
            # WARNING: error if a param is never optimized.
            assert len(param_groups[lookup[name]]['params']) == 1
            param_groups[lookup[name]]['params'][0] = new_param

            for k in self.state_attr_names:
                ori_state = state_dict[k]
                subdiv_state = ori_state[subdivide_idx].repeat_interleave(8, dim=0)
                state_dict[k] = mask_cat_perm(
                    ori_state,
                    kept_idx=kept_idx,
                    cat_tensor=subdiv_state)
                assert state_dict[k].shape == new_param.shape
                del ori_state, subdiv_state
                torch.cuda.empty_cache()
            state[new_param] = state_dict

            del new_param
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _prune_grid_pts_parameters(self, kept_idx):
        '''
        Prune trainable grid_pts parameters.
        NOTE: This function assume per-voxel attributes are updated.
        Input:
            @kept_idx: the voxels to be kept.
        '''
        # Shortcut to access and manipulate optimizer
        if hasattr(self, 'optimizer'):
            state = self.optimizer.state
            param_groups = self.optimizer.param_groups
            lookup = {
                group['name']: idx
                for idx, group in enumerate(param_groups)
            }
        else:
            state = None

        # Assume per-voxel attribute is updated.
        # Re-build the link between voxel and grid_pts.
        old_vox_key = self.vox_key.clone()
        new_grid_pts_key, new_vox_key = octree_utils.build_grid_pts_link(self.octpath, self.octlevel)
        new_num_grid_pts = len(new_grid_pts_key)
        self.grid_pts_key, self.vox_key = new_grid_pts_key, new_vox_key

        # Update grid_pts parameters from voxel
        for name in self.grid_pts_param_lst:
            param = getattr(self, name)
            if state is not None:
                state_dict = state.pop(param)
            ori_grid_pts = param.detach()
            del param
            torch.cuda.empty_cache()

            # Update parameter
            ori_vox_grid_pts_val = ori_grid_pts[old_vox_key]
            new_vox_val = mask_cat_perm(
                ori_vox_grid_pts_val,
                kept_idx=kept_idx)
            new_param = agg_voxel_into_grid_pts(
                new_num_grid_pts,
                new_vox_key,
                new_vox_val).requires_grad_()
            setattr(self, name, new_param)
            del ori_grid_pts, ori_vox_grid_pts_val, new_vox_val
            torch.cuda.empty_cache()

            # Update optimizer
            if state is not None:
                # WARNING: error if a param is never optimized.
                assert len(param_groups[lookup[name]]['params']) == 1
                param_groups[lookup[name]]['params'][0] = new_param

                for k in self.state_attr_names:
                    ori_grid_pts_state = state_dict[k]
                    ori_vox_grid_pts_state = ori_grid_pts_state[old_vox_key]
                    new_vox_state = mask_cat_perm(
                        ori_vox_grid_pts_state,
                        kept_idx=kept_idx)
                    new_state = agg_voxel_into_grid_pts(
                        new_num_grid_pts,
                        new_vox_key,
                        new_vox_state)
                    state_dict[k] = new_state
                    assert new_state.shape == new_param.shape
                    del ori_grid_pts_state, ori_vox_grid_pts_state, new_vox_state, new_state
                    torch.cuda.empty_cache()
                state[new_param] = state_dict

            del new_param
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _subdivide_grid_pts_parameters(self, kept_idx, subdivide_idx, save_gpu=False):
        '''
        Subdivide trainable grid_pts parameters.
        NOTE: This function assume per-voxel attributes are updated.
        Input:
            @kept_idx: the voxels to be kept.
            @subdivide_idx: the voxels to be subdivided.
        '''
        # Shortcut to access and manipulate optimizer
        state = self.optimizer.state
        param_groups = self.optimizer.param_groups
        lookup = {
            group['name']: idx
            for idx, group in enumerate(param_groups)
        }

        # Assume per-voxel attribute is updated.
        # Re-build the link between voxel and grid_pts.
        old_vox_key = self.vox_key.clone()
        new_grid_pts_key, new_vox_key = octree_utils.build_grid_pts_link(self.octpath, self.octlevel)
        new_num_grid_pts = len(new_grid_pts_key)
        self.grid_pts_key, self.vox_key = new_grid_pts_key, new_vox_key

        if save_gpu:
            kept_idx = kept_idx.cpu()
            subdivide_idx = subdivide_idx.cpu()
            old_vox_key = old_vox_key.cpu()

        # Update grid points parameters from voxel
        for name in self.grid_pts_param_lst:
            param = getattr(self, name)
            state_dict = state.pop(param)
            ori_grid_pts = param.detach()
            if save_gpu:
                ori_grid_pts = ori_grid_pts.cpu()
            del param
            torch.cuda.empty_cache()

            # Update parameter
            # First we gather grid_pts values into each voxel first.
            # The voxel is then subdivided by trilinear interpolation.
            # Finally, we gather voxel values back to the grid_pts.
            ori_vox_grid_pts_val = ori_grid_pts[old_vox_key]
            subdiv_vox_grid_pts_val = subdivide_by_interp(
                ori_vox_grid_pts_val[subdivide_idx])
            new_vox_val = mask_cat_perm(
                ori_vox_grid_pts_val,
                kept_idx=kept_idx,
                cat_tensor=subdiv_vox_grid_pts_val)
            new_param = agg_voxel_into_grid_pts(
                new_num_grid_pts,
                new_vox_key,
                new_vox_val).cuda().requires_grad_()
            setattr(self, name, new_param)
            del ori_grid_pts, ori_vox_grid_pts_val, subdiv_vox_grid_pts_val, new_vox_val
            torch.cuda.empty_cache()

            # Update optimizer
            # We do the same thing as parameter update.
            # WARNING: error if a param is never optimized.
            assert len(param_groups[lookup[name]]['params']) == 1
            param_groups[lookup[name]]['params'][0] = new_param

            for k in self.state_attr_names:
                ori_grid_pts_state = state_dict[k]
                if save_gpu:
                    ori_grid_pts_state = ori_grid_pts_state.cpu()
                ori_vox_grid_pts_state = ori_grid_pts_state[old_vox_key]
                subdiv_vox_grid_pts_state = subdivide_by_interp(
                    ori_vox_grid_pts_state[subdivide_idx])
                new_vox_state = mask_cat_perm(
                    ori_vox_grid_pts_state,
                    kept_idx=kept_idx,
                    cat_tensor=subdiv_vox_grid_pts_state)
                new_state = agg_voxel_into_grid_pts(
                    new_num_grid_pts,
                    new_vox_key,
                    new_vox_state)
                state_dict[k] = new_state.cuda()
                assert new_state.shape == new_param.shape
                del ori_grid_pts_state, ori_vox_grid_pts_state, subdiv_vox_grid_pts_state, new_vox_state, new_state
                torch.cuda.empty_cache()
            state[new_param] = state_dict

            del new_param
            torch.cuda.empty_cache()


# Some helpful functions
def mask_cat_perm(tensor, kept_idx=None, cat_tensor=None, perm=None):
    '''
    Perform tensor masking, concatenation, and permutation.
    '''
    if kept_idx is None and cat_tensor is None and perm is None:
        raise Exception("No op for mask_cat_perm??")
    device = tensor.device
    if kept_idx is not None:
        tensor = tensor[kept_idx.to(device)]
    if cat_tensor is not None:
        tensor = torch.cat([tensor, cat_tensor.to(device)])
    if perm is not None:
        assert len(perm) == len(tensor)
        tensor = tensor[perm.to(device)]
    return tensor.contiguous()

def agg_voxel_into_grid_pts(num_grid_pts, vox_key, vox_val, reduce='mean'):
    '''
    Aggregate per-voxel data into their eight grid points.
    Input:
        @num_grid_pts  Number of final grid points.
        @vox_key       [N, 8] Index to the eight grid points of each voxel.
        @vox_val       [N, 8, *] Data of the eight grid points of each voxel.
    Output:
        @new_param     [num_grid_pts, *] Grid points data aggregated from vox_val.
    '''
    ch = vox_val.shape[2:]
    device = vox_val.device
    vox_key = vox_key.to(device)
    new_param = torch.zeros([num_grid_pts, *ch], dtype=torch.float32, device=device)
    new_param.index_reduce_(
        dim=0,
        index=vox_key.flatten(),
        source=vox_val.flatten(0,1),
        reduce=reduce,
        include_self=False)
    # Equivalent implementation by old API
    # new_param /= vox_key.flatten().bincount(minlength=num_grid_pts).unsqueeze(-1)
    # new_param.nan_to_num_()
    return new_param.contiguous()

def subdivide_by_interp(vox_val):
    '''
    Subdivide grid point data by trilinear interpolation.
    The subdivided children order is the same as those from `_subdivide_attr` and `gen_children`.
    Input:
        @vox_val       [N, 8, *] Data of the eight grid points of each voxel.
    Output:
        @new_vox_val   [8N, 8, *] Data of the eight grid points of the subdivided voxel.
    '''
    vox_val = vox_val.contiguous()
    main_idx = torch.arange(8, dtype=torch.int64, device=vox_val.device)
    new_vox_val = torch.zeros([len(vox_val), 8, *vox_val.shape[1:]], device=vox_val.device)
    new_vox_val[:, main_idx, main_idx] = vox_val
    new_vox_val[:, main_idx, main_idx^0b001] = 0.5 * (vox_val + vox_val[:, main_idx^0b001])
    new_vox_val[:, main_idx, main_idx^0b010] = 0.5 * (vox_val + vox_val[:, main_idx^0b010])
    new_vox_val[:, main_idx, main_idx^0b100] = 0.5 * (vox_val + vox_val[:, main_idx^0b100])
    new_vox_val[:, main_idx, main_idx^0b011] = 0.25 * (
        vox_val + \
        vox_val[:, main_idx^0b001] + \
        vox_val[:, main_idx^0b010] + \
        vox_val[:, main_idx^0b011]
    )
    new_vox_val[:, main_idx, main_idx^0b101] = 0.25 * (
        vox_val + \
        vox_val[:, main_idx^0b001] + \
        vox_val[:, main_idx^0b100] + \
        vox_val[:, main_idx^0b101]
    )
    new_vox_val[:, main_idx, main_idx^0b110] = 0.25 * (
        vox_val + \
        vox_val[:, main_idx^0b010] + \
        vox_val[:, main_idx^0b100] + \
        vox_val[:, main_idx^0b110]
    )
    new_vox_val[:, main_idx, main_idx^0b111] = vox_val.mean(1, keepdim=True)

    new_vox_val = new_vox_val.reshape(len(vox_val)*8, *vox_val.shape[1:])
    return new_vox_val.contiguous()
