/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(
		const dim3 grid,
		const int P, int D, int M,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		int* radii,
		bool* clamped,
		float3* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* cov_opacity,
		float4* lambda_sigma,
		float4* nv1_nv2,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const int W, int H,
		const float* bg_color,
		const float* features,
		const uint32_t* point_list,
		const float3* points_xy_image,
		const float4* cov_opacity,
		const float4* lambda_sigma,
		const float4* nv1_nv2,
		const uint2* ranges,
		float* final_T,
		uint32_t* n_contrib,
		float* out_color);
}


#endif
