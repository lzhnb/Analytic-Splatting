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

#include "forward.h"
#include "vec_math.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(
	const int idx,
	const int deg,
	const int max_coeffs,
	const glm::vec3* means,
	const glm::vec3 campos,
	const float* shs,
	bool* clamped
) {
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0) {
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1) {
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2) {
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(
	const float3& mean,
	float focal_x,
	float focal_y,
	float tan_fovx,
	float tan_fovy,
	const float* cov3D,
	const float* viewmatrix
) {
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// NOTE: we only care the projected 2D cov, so we can ignore the third row in J
	// EWA splatting implement the standard camera coordinate in Eq. (29)
	// And we use the focal to scale the mat and make it suitable for our camera model
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0.0f, 0.0f, 0.0f);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(
	const glm::vec3 scale,
	float mod,
	const glm::vec4 rot,
	float* cov3D
) {
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
	const dim3 grid,
	const int P, int D, int M,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	const float* __restrict__ orig_points,
	const glm::vec3* __restrict__ scales,
	const float scale_modifier,
	const glm::vec4* __restrict__ rotations,
	const float* __restrict__ opacities,
	const float* __restrict__ shs,
	const float* __restrict__ cov3D_precomp,
	const float* __restrict__ colors_precomp,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix,
	const glm::vec3* __restrict__ cam_pos,
	int* __restrict__ radii,
	bool* __restrict__ clamped,
	float3* __restrict__ points_xy_image,
	float* __restrict__ depths,
	float* __restrict__ cov3Ds,
	float* __restrict__ rgb,
	float4* __restrict__ cov_opacity,
	float4* __restrict__ lambda_sigma,
	float4* __restrict__ nv1_nv2,
	uint32_t* __restrict__ tiles_touched,
	bool prefiltered
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	const float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	const float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	const float p_w = 1.0f / (p_hom.w + 0.0000001f);
	const float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr) {
		cov3D = cov3D_precomp + idx * 6;
	} else {
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	const float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	const float det = cov.x * cov.z - cov.y * cov.y;
	if (det == 0.0f)
		return;

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	const float half_trace = 0.5f * (cov.x + cov.z);
	const float root = half_trace * half_trace - det;
	const float root_sqrt = sqrtf(max(0.0f, root));
	const float lambda1 = half_trace + root_sqrt;
	const float lambda2 = half_trace - root_sqrt;
	// const float lambda1_ = half_trace + max(0.33f, root_sqrt);
	// const float lambda2_ = half_trace - max(0.33f, root_sqrt);
	const float lambda1_ = half_trace + sqrt(max(0.1f, root));
	const float lambda2_ = half_trace - sqrt(max(0.1f, root));
	const float my_radius = ceil(3.f * sqrtf(max(lambda1_, lambda2_)));
	const float3 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H), root_sqrt};
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// https://math.stackexchange.com/questions/395698/fast-way-to-calculate-eigen-of-2x2-matrix-using-a-formula
	float2 v1 = {cov.y, lambda1 - cov.x};
	v1 = normalize(v1);
	float2 v2 = {lambda2 - cov.z, cov.y};
	v2 = normalize(v2);
	const float sigma1 = sqrtf(lambda1), sigma2 = sqrtf(abs(lambda2));

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	cov_opacity[idx] = { cov.x, cov.y, cov.z, opacities[idx] };
	lambda_sigma[idx] = {lambda1, lambda2, sigma1, sigma2};
	nv1_nv2[idx] = {v1.x, v1.y, v2.x, v2.y};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const int W, int H,
	const float* __restrict__ bg_color,
	const float* __restrict__ features,
	const uint32_t* __restrict__ point_list,
	const float3* __restrict__ points_xy_image,
	const float4* __restrict__ cov_opacity,
	const float4* __restrict__ lambda_sigma,
	const float4* __restrict__ nv1_nv2,
	const uint2* __restrict__ ranges,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_color
) {
	// Identify current tile and associated min/max pixel range.
	const auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	const bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_cov_opacity[BLOCK_SIZE];
	__shared__ float4 collected_lambda_sigma[BLOCK_SIZE];
	__shared__ float4 collected_nv1_nv2[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0.0f };

	// Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) {
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_cov_opacity[block.thread_rank()] = cov_opacity[coll_id];
			collected_lambda_sigma[block.thread_rank()] = lambda_sigma[coll_id];
			collected_nv1_nv2[block.thread_rank()] = nv1_nv2[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
			// Keep track of current position in range
			contributor++;

			float3 xy = collected_xy[j];
			float2 d = { pixf.x - xy.x, pixf.y - xy.y };
			float4 cov_o = collected_cov_opacity[j];

			float2 v1 = {collected_nv1_nv2[j].x, collected_nv1_nv2[j].y}, v2 = {collected_nv1_nv2[j].z, collected_nv1_nv2[j].w};
			// // calculate the uv by projection
			float2 uv = {d.x * v1.x + d.y * v1.y, d.x * v2.x + d.y * v2.y};
			float sigma1 = collected_lambda_sigma[j].z, sigma2 = collected_lambda_sigma[j].w;

			// Equal to exp(power)
			// const float sigma1 = sqrtf(lambda1), sigma2 = sqrtf(abs(lambda2));
			const float U2 = (uv.x + 0.5f) / sigma1, U1 = (uv.x - 0.5f) / sigma1;
			const float cdfU2 = approxCdf(U2), cdfU1 = approxCdf(U1);
			const float intU = sigma1 * (cdfU2 - cdfU1);
			const float V2 = (uv.y + 0.5f) / sigma2, V1 = (uv.y - 0.5f) / sigma2;
			const float cdfV2 = approxCdf(V2), cdfV1 = approxCdf(V1);
			const float intV = sigma2 * (cdfV2 - cdfV1);
			const float integral = M_2PIf * intU * intV;

			float alpha = min(0.99f, cov_o.w * integral);

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			if (alpha < 1.0f / 255.0f) continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) {
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside) {
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const int W, int H,
	const float* bg_color,
	const float* colors,
	const uint32_t* point_list,
	const float3* means2D,
	const float4* cov_opacity,
	const float4* lambda_sigma,
	const float4* nv1_nv2,
	const uint2* ranges,
	float* final_T,
	uint32_t* n_contrib,
	float* out_color
) {
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		W, H,
		bg_color,
		colors,
		point_list,
		means2D,
		cov_opacity,
		lambda_sigma,
		nv1_nv2,
		ranges,
		final_T,
		n_contrib,
		out_color);
}

void FORWARD::preprocess(
	const dim3 grid,
	const int P, int D, int M,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float* means3D,
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
	float3* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* cov_opacity,
	float4* lambda_sigma,
	float4* nv1_nv2,
	uint32_t* tiles_touched,
	bool prefiltered
) {
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		grid,
		P, D, M,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		radii,
		clamped,
		means2D,
		depths,
		cov3Ds,
		rgb,
		cov_opacity,
		lambda_sigma,
		nv1_nv2,
		tiles_touched,
		prefiltered
	);
}
