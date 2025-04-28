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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* opacities,
	const float* dL_dconics,
	float* dL_dopacity,
	const float* dL_dinvdepth,
	float3* dL_dmeans,
	float* dL_dcov,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float c_xx = cov2D[0][0];
	float c_xy = cov2D[0][1];
	float c_yy = cov2D[1][1];
	
	constexpr float h_var = 0.3f;
	float d_inside_root = 0.f;
	if(antialiasing)
	{
		const float det_cov = c_xx * c_yy - c_xy * c_xy;
		c_xx += h_var;
		c_yy += h_var;
		const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
		const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
		const float dL_dopacity_v = dL_dopacity[idx];
		const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
		dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;
		d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
	} 
	else
	{
		c_xx += h_var;
		c_yy += h_var;
	}
	
	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	if(antialiasing)
	{
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
	}
	
	float denom = c_xx * c_yy - c_xy * c_xy;

	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		
		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xx +
	(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xx +
	(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xx +
	(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_xy;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_yy +
	(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_yy +
	(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_yy +
	(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xy;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;
	// Account for inverse depth gradients
	if (dL_dinvdepth)
	dL_dtz -= dL_dinvdepth[idx] / (t.z * t.z);


	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

template<int C>
__global__ void preprocessCUDA(
    int P, int D, int M,
    const float3* means3D,
    const int* radii,
    const float* shs,
    const bool* clamped,
    const glm::vec3* scales,
    const glm::vec3* skews,
    const float* skew_sensitivity,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3D,
    const float* viewmatrix,
    const float* projmatrix,
    const float focal_x, const float focal_y,
    const float tan_fovx, const float tan_fovy,
    const glm::vec3* campos,
    const float3* dL_dmean2D,
    const float4* dL_dconic2D,
    float3* dL_dmeans3D,
    float* dL_dcolor,
    float* dL_dcov3D,
    const float2* dL_dskews2D,
    float* dL_dskews,
    float* dL_dskew_sensitivity,
    float* dL_dsh,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot,
	float* dL_dopacity)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0)) return;

    float3 m = means3D[idx];
    float4 m_hom = transformPoint4x4(m, projmatrix);
    float m_w = 1.0f / (m_hom.w + 1e-7f);

    float mul1 = (projmatrix[0]*m.x + projmatrix[4]*m.y + projmatrix[8]*m.z + projmatrix[12]) * m_w * m_w;
    float mul2 = (projmatrix[1]*m.x + projmatrix[5]*m.y + projmatrix[9]*m.z + projmatrix[13]) * m_w * m_w;

    glm::vec3 dL_dmean;
    dL_dmean.x = (projmatrix[0]*m_w - projmatrix[3]*mul1) * dL_dmean2D[idx].x + (projmatrix[1]*m_w - projmatrix[3]*mul2) * dL_dmean2D[idx].y;
    dL_dmean.y = (projmatrix[4]*m_w - projmatrix[7]*mul1) * dL_dmean2D[idx].x + (projmatrix[5]*m_w - projmatrix[7]*mul2) * dL_dmean2D[idx].y;
    dL_dmean.z = (projmatrix[8]*m_w - projmatrix[11]*mul1) * dL_dmean2D[idx].x + (projmatrix[9]*m_w - projmatrix[11]*mul2) * dL_dmean2D[idx].y;
    dL_dmean = -dL_dmean;
	
	float3 tmp = dL_dmeans3D[idx];
	float3 delta = make_float3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
	tmp.x += delta.x;
	tmp.y += delta.y;
	tmp.z += delta.z;
	dL_dmeans3D[idx] = tmp;

    if (dL_dskews2D && dL_dskews) {
        float gradx = dL_dskews2D[idx].x;
        float grady = dL_dskews2D[idx].y;

        glm::vec3 dL_dskew_cam;
        dL_dskew_cam.x = (projmatrix[0]*m_w - projmatrix[3]*mul1) * gradx + (projmatrix[1]*m_w - projmatrix[3]*mul2) * grady;
        dL_dskew_cam.y = (projmatrix[4]*m_w - projmatrix[7]*mul1) * gradx + (projmatrix[5]*m_w - projmatrix[7]*mul2) * grady;
        dL_dskew_cam.z = (projmatrix[8]*m_w - projmatrix[11]*mul1)* gradx + (projmatrix[9]*m_w - projmatrix[11]*mul2)* grady;

        float3 dL_dskew_world = {
			viewmatrix[0]*dL_dskew_cam.x + viewmatrix[1]*dL_dskew_cam.y + viewmatrix[2]*dL_dskew_cam.z,
			viewmatrix[4]*dL_dskew_cam.x + viewmatrix[5]*dL_dskew_cam.y + viewmatrix[6]*dL_dskew_cam.z,
			viewmatrix[8]*dL_dskew_cam.x + viewmatrix[9]*dL_dskew_cam.y + viewmatrix[10]*dL_dskew_cam.z };
		

		auto* dL_dskews_f3 = reinterpret_cast<float3*>(dL_dskews);
        atomicAdd(&dL_dskews_f3[idx].x, dL_dskew_world.x);
        atomicAdd(&dL_dskews_f3[idx].y, dL_dskew_world.y);
        atomicAdd(&dL_dskews_f3[idx].z, dL_dskew_world.z);
    }

    if (shs) computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped,
                                 (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans3D, (glm::vec3*)dL_dsh);

    if (scales) computeCov3D(idx, scales[idx], scale_modifier, rotations[idx],
                              dL_dcov3D, dL_dscale, dL_drot);
}

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2*  __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int   W, int   H,
    const float*   __restrict__ bg_color,
    const float2*  __restrict__ points_xy_image,
    const float2*  __restrict__ skews2D,
    const float*   __restrict__ skew_sensitivity,
    const float4*  __restrict__ conic_opacity,
    const float*   __restrict__ colors,
	const float* 				depths,
    const float*   __restrict__ final_Ts,
    const uint32_t*__restrict__ n_contrib,
    const float*   __restrict__ dL_dpixels,
	const float* 				dL_invdepths,
          float3*  __restrict__ dL_dmean2D,
          float4*  __restrict__ dL_dconic2D,
          float2*  __restrict__ dL_dskews2D,
          float*   __restrict__ dL_dskew_sensitivity,
          float*   __restrict__ dL_dopacity,
		  float* __restrict__ dL_dcolors,
		  float* __restrict__ dL_dinvdepths)
{
    // Configuración del bloque y determinación de posición del píxel
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };

    const bool inside = pix.x < W && pix.y < H;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bool done = !inside;
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

    // En el forward, almacenamos el valor final para T,
    // el producto de todos los factores (1 - alpha).
    const float T_final = inside ? final_Ts[pix_id] : 0;
    float T = T_final;

    // Comenzamos desde atrás. El ID de la última Gaussiana contribuyente
    // es conocido desde cada píxel en el forward.
    uint32_t contributor = toDo;
    const int last_contributor = inside ? n_contrib[pix_id] : 0;

    float accum_rec[C] = { 0 };
    float dL_dpixel[C];
	float dL_invdepth;
	float accum_invdepth_rec = 0;

    if (inside)
	{
        for (int i = 0; i < C; i++)
            dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		if(dL_invdepths)
			dL_invdepth = dL_invdepths[pix_id];
	}
    float last_alpha = 0;
    float last_color[C] = { 0 };
	float last_invdepth = 0;
    // Gradiente de coordenada de píxel con respecto a 
    // coordenadas de viewport normalizadas (-1 a 1)
    const float ddelx_dx = 0.5f * W;
    const float ddely_dy = 0.5f * H;

    // Recorrer todas las Gaussianas
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // Cargar datos auxiliares en memoria compartida, comenzando desde ATRÁS
        // y cargándolos en orden inverso.
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            const int coll_id = point_list[range.y - progress - 1];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            for (int i = 0; i < C; i++)
                collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];

			if(dL_invdepths)
				collected_depths[block.thread_rank()] = depths[coll_id];
        }
        block.sync();

        // Iterar sobre Gaussianas
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            // Seguir la pista del ID de la Gaussiana actual. Omitir, si ésta
            // está detrás del último contribuyente para este píxel.
            contributor--;
            if (contributor >= last_contributor)
                continue;

            // Calcular valores de mezcla, como antes.
            const float2 xy = collected_xy[j];
            const float2 d  = { xy.x - pixf.x, xy.y - pixf.y };
            const float4 C4 = collected_conic_opacity[j];

            const float G  = __expf(-0.5f*(C4.x*d.x*d.x + C4.z*d.y*d.y) - C4.y*d.x*d.y);
            const float A_raw = C4.w * G;
            const float A     = fminf(0.99f, A_raw);
            const bool  h     = (A_raw <= 0.99f);

            const uint  gid   = collected_id[j];
            const float2 s2D  = skews2D[gid];
            const float2 dB   = { d.x - s2D.x, d.y - s2D.y };
            const float  B    = C4.w * __expf(-0.5f*(C4.x*dB.x*dB.x + C4.z*dB.y*dB.y) - C4.y*dB.x*dB.y);

            const float  S    = skew_sensitivity[gid];
            const float  mask = 1.f - __expf(-S * B);
            const float  alpha= A * mask;
            if (alpha < 1.f/255.f) continue;

            /* ---------- compositing ---------- */
            T = T / (1.f - alpha);

            /* ---------- dL/dα acumulado ---------- */
            float dL_dalpha = 0.f;
			const uint global_id = collected_id[j];
			const float dchannel_dcolor = alpha * T;
            for (int ch = 0; ch < C; ++ch) 
            {
                const float c = collected_colors[ch * BLOCK_SIZE + j];
                // Actualizar último color (a usar en la próxima iteración)
                accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
                last_color[ch] = c;

                const float dL_dchannel = dL_dpixel[ch];
                dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
                // Actualizar los gradientes con respecto al color de la Gaussiana.
                // Atómico, ya que este píxel es solo uno de potencialmente
                // muchos que fueron afectados por esta Gaussiana.
                atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
            }
			if (dL_dinvdepths)
			{
				const float invd = 1.f / collected_depths[j];
				accum_invdepth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_invdepth_rec;
				last_invdepth = invd;
				dL_dalpha += (invd - accum_invdepth_rec) * dL_invdepth;
				atomicAdd(&(dL_dinvdepths[global_id]), dchannel_dcolor * dL_invdepth);
			}
            dL_dalpha *= T;
            
            // Tener en cuenta que alpha también influye en cuánto del
            // color de fondo se agrega si no queda nada para mezclar
            float bg_dot_dpixel = 0;
            for (int i = 0; i < C; i++)
                bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
            dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
            
             /* ===== DERIVADAS ANALÍTICAS ===== */
			 const float dalpha_dA = h ? mask : 0.f;        // h·m
			 const float dalpha_dB = A * S * (1.f - mask);  // A S (1-m)
			 const float dalpha_dS = A * B * (1.f - mask);  // A B (1-m)
 
			 // -- sensitividad S
			 atomicAdd(&dL_dskew_sensitivity[gid], dL_dalpha * dalpha_dS);
 
			 // -- conic (cxx,cxy,cyy)  [-½ A (…) ]
			 const float3 qA = { d.x*d.x,      2.f*d.x*d.y,      d.y*d.y };
			 const float3 qB = { dB.x*dB.x,    2.f*dB.x*dB.y,    dB.y*dB.y };
			 const float common = -0.5f * dL_dalpha * A;
 
			 atomicAdd(&dL_dconic2D[gid].x, common * (dalpha_dA * qA.x + dalpha_dB * qB.x));
			 atomicAdd(&dL_dconic2D[gid].y, common * (dalpha_dA * qA.y + dalpha_dB * qB.y));
			 atomicAdd(&dL_dconic2D[gid].w, common * (dalpha_dA * qA.z + dalpha_dB * qB.z));
 
			 // -- opacidad base o
			 float dL_do = dL_dalpha * ( h ? (dalpha_dA * G) : 0.f
                            +  dalpha_dB * (B / C4.w) );
			 atomicAdd(&dL_dopacity[gid], dL_do);

			
			 // -- skew-pix
			 const float2 C_dB = { C4.x*dB.x + C4.y*dB.y,
								   C4.y*dB.x + C4.z*dB.y };
			
			 float2 gradS = make_float2(dalpha_dB * B, dalpha_dB * B);
			 gradS.x *= C_dB.x;  gradS.y *= C_dB.y;
			 atomicAdd(&dL_dskews2D[gid].x, dL_dalpha * gradS.x);
			 atomicAdd(&dL_dskews2D[gid].y, dL_dalpha * gradS.y);
 
			 // -- mean2D
			 const float2 C_d  = { C4.x*d.x + C4.y*d.y,
								   C4.y*d.x + C4.z*d.y };
			 float2 g = { dalpha_dA * C_d.x + dalpha_dB * B * C_dB.x,
  	 						dalpha_dA * C_d.y + dalpha_dB * B * C_dB.y };
 
			 atomicAdd(&dL_dmean2D[gid].x, dL_dalpha * A * g.x * ddelx_dx);
			 atomicAdd(&dL_dmean2D[gid].y, dL_dalpha * A * g.y * ddely_dy);
 
			 last_alpha = alpha;
        }
    }
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
        printf("CUDA-ERROR %s %s:%d\n", cudaGetErrorString(code), file, line);
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float* opacities,
	const glm::vec3* scales,
	const glm::vec3* skews,
	const float* skew_sensitivity,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic2D,
	const float* dL_dinvdepth,
	float* dL_dopacity,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float2* dL_dskews2D,
	float* dL_dskews,
	float* dL_dskew_sensitivity,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	bool antialiasing)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		opacities,
		dL_dconic2D,
		dL_dopacity,
		dL_dinvdepth,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		antialiasing);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec3*)skews,
		skew_sensitivity,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3Ds,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		campos,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic2D,
		(float3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dskews2D,
		dL_dskews,
		dL_dskew_sensitivity,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dopacity);
		//CUDA_CHECK( cudaGetLastError() );
		//CUDA_CHECK( cudaDeviceSynchronize() );
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float2* skews2D,
	const float* skew_sensitivity,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_invdepths,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float2* dL_dskews2D,
	float* dL_dskew_sensitivity,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dinvdepths)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		skews2D,
		skew_sensitivity,
		conic_opacity,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_invdepths,
		dL_dmean2D,
		dL_dconic2D,
		dL_dskews2D,
		dL_dskew_sensitivity,
		dL_dopacity,
		dL_dcolors,
		dL_dinvdepths
		);
		//CUDA_CHECK( cudaGetLastError() );
		//CUDA_CHECK( cudaDeviceSynchronize() );
}