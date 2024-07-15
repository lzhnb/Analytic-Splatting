#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import torch
import torchvision
from lpips import LPIPS
from tqdm import tqdm

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Camera, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr as get_psnr
from utils.loss_utils import ssim as get_ssim

lpips_fn = LPIPS(net="vgg").cuda()


def render_set(
    model_path: str,
    name: str,
    iteration: int,
    views: List[Camera],
    gaussians: GaussianModel,
    pipeline: GroupParams,
    background: torch.Tensor,
    interval: int = 5,
    lpips: bool = False,
    vis: bool = False,
    filter3d: bool = False,
) -> None:
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")

    os.makedirs(render_path, exist_ok=True)

    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # for idx, view in enumerate(views):
        rendering = render(view, gaussians, pipeline, background, filter3d=filter3d)["render"]
        gt = view.original_image[0:3, :, :]
        if vis and idx % interval == 0:
            img = torch.cat([rendering, gt], dim=-1)
            torchvision.utils.save_image(img, os.path.join(render_path, f"{idx:05d}" + ".png"))
        psnr_avg += get_psnr(gt, rendering).mean().double()
        ssim_avg += get_ssim(gt, rendering).mean().double()
        if lpips:
            lpips_avg += lpips_fn(gt, rendering).mean().double()

    psnr = psnr_avg / len(views)
    ssim = ssim_avg / len(views)
    lpips = lpips_avg / len(views)
    content = f"psnr_avg: {psnr:.4f}; ssim_avg: {ssim:.4f}; lpips_avg: {lpips:.5f}"
    print(content)

    with open(os.path.join(model_path, name, f"ours_{iteration}", "results.txt"), "w") as fp:
        fp.write(content)


@torch.no_grad()
def launch(
    dataset: ModelParams,
    pipeline: PipelineParams,
    iteration: int,
    skip_train: bool,
    skip_test: bool,
    lpips: bool = False,
    vis: bool = False,
    filter3d: bool = False,
    interval: int = 5,
) -> None:
    gaussians = GaussianModel(sh_degree=dataset.sh_degree)
    scales = [1.0, 2.0, 4.0, 8.0]
    scene = Scene(
        args=dataset,
        load_iteration=iteration,
        gaussians=gaussians,
        shuffle=False,
        resolution_scales=scales,
    )
    # gaussians.restore(model_params)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        for scale in scales:
            print(f"eval train scale-{scale}")
            trainCameras = scene.getTrainCameras(scale).copy()
            render_set(
                model_path=dataset.model_path,
                name=f"train_{scale}",
                iteration=iteration,
                views=trainCameras,
                gaussians=gaussians,
                pipeline=pipeline,
                background=background,
                lpips=lpips,
                vis=vis,
                filter3d=filter3d,
                interval=interval,
            )

    if not skip_test:
        for scale in scales:
            print(f"eval test scale-{scale}")
            testCameras = scene.getTestCameras(scale).copy()
            render_set(
                model_path=dataset.model_path,
                name=f"test_{scale}",
                iteration=iteration,
                views=testCameras,
                gaussians=gaussians,
                pipeline=pipeline,
                background=background,
                lpips=lpips,
                vis=vis,
                filter3d=filter3d,
                interval=interval,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--interval", default=5, type=int)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ndown", type=int, default=-1)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter3d", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    launch(
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        iteration=args.iteration,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        lpips=args.lpips,
        vis=args.vis,
        filter3d=args.filter3d,
        interval=args.interval,
    )
