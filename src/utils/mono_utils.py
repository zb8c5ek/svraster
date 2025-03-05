# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import cv2
import tqdm
import torch
import numpy as np
import imageio.v2 as iio
from PIL import Image


@torch.no_grad()
def prepare_depthanythingv2(cameras, source_path, force_rerun=False):

    depth_root = os.path.join(source_path, "mono_priors", "depthanythingv2")
    os.makedirs(depth_root, exist_ok=True)

    depth_path = lambda cam: os.path.join(depth_root, f"{cam.image_name}.png")
    codebook_path = lambda cam: os.path.join(depth_root, f"{cam.image_name}.npy")

    # Inference estimated depths if not done before
    todo_indices = []
    for i, cam in enumerate(cameras):
        if not os.path.exists(depth_path(cam)) or force_rerun:
            todo_indices.append(i)
    
    if len(todo_indices):
        print(f"Infer depth for {len(todo_indices)} images. Saved to {depth_root}.")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
        model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").cuda()

    for i in tqdm.tqdm(todo_indices):
        cam = cameras[i]

        # Inference depth
        inputs = image_processor(images=cam.image, return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = model(**inputs)
        depth = outputs['predicted_depth'].squeeze()

        # Quantization
        codebook = depth.quantile(torch.linspace(0, 1, 65536).cuda(), interpolation='nearest')
        depth_idx = torch.searchsorted(codebook, depth, side='right').clamp_max_(65535)
        depth_idx[(depth - codebook[depth_idx-1]).abs() < (depth - codebook[depth_idx]).abs()] -= 1
        assert depth_idx.max() <= 65535
        assert depth_idx.min() >= 0

        # Save result
        depth_np = depth_idx.cpu().numpy().astype(np.uint16)
        iio.imwrite(depth_path(cam), depth_np)
        np.save(codebook_path(cam), codebook.cpu().numpy().astype(np.float32))

    # Load the estimated depth
    print("Load the estimated depths to cameras.")
    for cam in tqdm.tqdm(cameras):
        depth_np = iio.imread(depth_path(cam))
        codebook = np.load(codebook_path(cam))
        cam.depthanthingv2 = torch.tensor(codebook[depth_np])
