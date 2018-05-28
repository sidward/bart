/* Copyright 2018. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2018 Siddharth Iyer <ssi@mit.edu>
 *
 * Tamir J, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M. 
 * T2 shuffling: Sharp, multicontrast, volumetric fast spin‐echo imaging. 
 * Magnetic resonance in medicine. 2017 Jan 1;77(1):180-95.
 *
 * B Bilgic, BA Gagoski, SF Cauley, AP Fan, JR Polimeni, PE Grant,
 * LL Wald, and K Setsompop, Wave-CAIPI for highly accelerated 3D
 * imaging. Magn Reson Med (2014) doi: 10.1002/mrm.25347
 *
 * Iyer S, Bilgic B, Setsompop K.
 * Faster T2 shuffling with Wave.
 * Submitted to ISMRM 2018.
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/iovec.h"

#include "iter/iter.h"
#include "iter/lsqr.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "wavelet/wavthresh.h"
#include "lowrank/lrthresh.h"

static const char usage_str[] = "<maps> <wave> <phi> <reorder> <data> <output>";
static const char help_str[] = "Perform wave-shuffling reconstruction.\n\n"
															 "Conventions:\n"
															 "	* (sx, sy, sz) - Spatial dimensions.\n"
															 "	* wx					 - Extended FOV in READ_DIM due to\n"
															 "									 wave's voxel spreading.\n"
															 "	* (nc, md)		 - Number of channels and ESPIRiT's \n"
															 "									 extended-SENSE model operator\n"
															 "									 dimensions (or # of maps).\n"
															 "	* (tf, tk)		 - Turbo-factor and the rank\n"
															 "									 of the temporal basis used in\n"
															 "									 shuffling.\n"
															 "	* ntr					 - Number of TRs, or the number of\n"
															 "									 (ky, kz) points acquired of one\n"
															 "									 echo image.\n"
															 "	* n						 - Total number of (ky, kz) points\n"
															 "									 acquired. This is equal to the\n"
															 "									 product of ntr and tf.\n\n"
															 "Descriptions:\n"
															 "	* reorder is an (n by 3) index matrix such that\n"
															 "		[ky, kz, t] = reorder(i, :) represents the\n"
															 "		(ky, kz) kspace position of the readout line\n" 
															 "		acquired at echo number (t), and 0 <= ky < sy,\n"
															 "		0 <= kz < sz, 0 <= t < tf).\n"
															 "	* data is a (wx by nc by n) matrix such that\n"
															 "		data(:, :, k) represents the kth multichannel\n"
															 "		kspace line.\n\n"
															 "Expected dimensions:\n"
															 "	* maps		- (		sx, sy, sz, nc, md,  1,  1)\n"
															 "	* wave		- (		wx, sy, sz,  1,  1,  1,  1)\n"
															 "	* phi			- (		 1,  1,  1,  1,  1, tf, tk)\n"
															 "	* output	- (		sx, sy, sz,  1, md,  1, tk)\n"
															 "	* reorder - (		 n,  3,  1,  1,  1,  1,  1)\n"
															 "	* data		- (		wx, nc,  n,  1,  1,  1,  1)";

/* Helper function to print out operator dimensions. */
static void print_opdims(const struct linop_s* op) 
{
	const struct iovec_s* domain	 = linop_domain(op);
	const struct iovec_s* codomain = linop_codomain(op);
	debug_printf(DP_INFO, "  domain: ");
	debug_print_dims(DP_INFO, domain->N, domain->dims);
	debug_printf(DP_INFO, "codomain: ");
	debug_print_dims(DP_INFO, codomain->N, codomain->dims);
}

/* Construct sampling mask from reorder tables. */
static void construct_mask(long reorder_dims[DIMS], complex float* reorder, 
													 long mask_dims[DIMS],		complex float* mask)
{
	int n  = reorder_dims[0];
	int sy = mask_dims[1];
	int sz = mask_dims[2];

	int y = -1;
	int z = -1;
	int t = -1;
	
	for (int i = 0; i < n; i++) {
		y = reorder[i];
		z = reorder[i + n];
		t = reorder[i + 2 * n];
		mask[(y + z * sy) + t * sy * sz] = 1;
	}
}

/* Collapse table into the temporal basis for memory efficiency. */
static void collapse_table(long reorder_dims[DIMS],  complex float* reorder,
													 long phi_dims[DIMS],			 complex float* phi, 
													 long data_dims[DIMS],		 complex float* data,
													 long collapse_dims[DIMS], complex float* collapse)
{
	long wx = data_dims[0];
	long sy = collapse_dims[1];
	long sz = collapse_dims[2];
	long nc = data_dims[1];
	long n	= reorder_dims[0];
	long tf = phi_dims[5];
	long tk = phi_dims[6];
	int  t	= -1;

	long vec_dims[]			= {wx * nc, tf,  1};
	long phi_mat_dims[] = {			 1, tf, tk};
	long phi_out_dims[] = {wx * nc,  1, tk};
	long fmac_dims[]		= {wx * nc, tf, tk};

	long out_dims[]			= { [0 ... DIMS - 1] = 1 };
	out_dims[0]					= wx;
	out_dims[1]					= nc;
	out_dims[2]					= tk;
	out_dims[3]					= sy;
	out_dims[4]					= sz;
	long copy_dim[]			= {wx * nc};

	complex float* vec = md_alloc_sameplace(	 3, vec_dims, CFL_SIZE, collapse);
	complex float* out = md_alloc_sameplace(DIMS, out_dims, CFL_SIZE, collapse);

	long vec_str[3];
	md_calc_strides(3, vec_str, vec_dims, CFL_SIZE);

	long phi_mat_str[3];
	md_calc_strides(3, phi_mat_str, phi_mat_dims, CFL_SIZE);

	long phi_out_str[3];
	md_calc_strides(3, phi_out_str, phi_out_dims, CFL_SIZE);

	long fmac_str[3];
	md_calc_strides(3, fmac_str, fmac_dims, CFL_SIZE);

	for (int ky = 0; ky < sy; ky ++) {
		for (int kz = 0; kz < sz; kz ++) {

			md_clear(3, vec_dims, vec, CFL_SIZE);

			for (int i = 0; i < n; i ++) {
				if ((ky == reorder[i]) && (kz == reorder[i + n])) {
					t = reorder[i + 2 * n];
					md_copy(1, copy_dim, (vec + t * wx * nc), (data + i * wx * nc), CFL_SIZE);
				}
			}

			md_zfmacc2(3, fmac_dims, phi_out_str, (out + (kz * sy + ky) * (wx * nc * tk)), vec_str, vec, 
				phi_mat_str, phi);

		}
	}

	unsigned int permute_order[DIMS] = {0, 3, 4, 1, 5, 6, 2, 7};
	for (unsigned int i = 8; i < DIMS; i++)
		permute_order[i] = i;
	md_permute(DIMS, permute_order, collapse_dims, collapse, out_dims, out, CFL_SIZE);

	md_free(vec);
	md_free(out);
}

/* ESPIRiT operator. */
static void E(long input_dims[DIMS], const complex float* input,
							long maps_dims[DIMS],  const complex float* maps, 
							bool adj,
							long out_dims[DIMS],				 complex float* out)
{
	long input_str[DIMS];
	md_calc_strides(DIMS, input_str, input_dims, CFL_SIZE);

	long maps_str[DIMS];
	md_calc_strides(DIMS, maps_str, maps_dims, CFL_SIZE);

	long fmac_dims[DIMS];
	md_merge_dims(DIMS, fmac_dims, input_dims, maps_dims);
	long fmac_str[DIMS];
	md_calc_strides(DIMS, fmac_str, fmac_dims, CFL_SIZE);

	unsigned long squash = (adj ? 8 : 16);
	md_select_dims(DIMS, ~squash, out_dims, fmac_dims);
	long out_str[DIMS];
	md_calc_strides(DIMS, out_str, out_dims, CFL_SIZE);

	(adj ? md_zfmacc2 : md_zfmac2)(DIMS, fmac_dims, out_str, out, input_str, input, maps_str, maps);
}

/* Shuffling temporal operator. */
static void P(long input_dims[DIMS], const complex float* input,
							long phi_dims[DIMS],	 const complex float* phi, 
							bool adj,
							long out_dims[DIMS],				 complex float* out)
{
	long input_str[DIMS];
	md_calc_strides(DIMS, input_str, input_dims, CFL_SIZE);

	long phi_str[DIMS];
	md_calc_strides(DIMS, phi_str, phi_dims, CFL_SIZE);

	long fmac_dims[DIMS];
	md_merge_dims(DIMS, fmac_dims, input_dims, phi_dims);

	unsigned long squash = (adj ? 32 : 64);
	md_select_dims(DIMS, ~squash, out_dims, fmac_dims);
	long out_str[DIMS];
	md_calc_strides(DIMS, out_str, out_dims, CFL_SIZE);

	(adj ? md_zfmacc2 : md_zfmac2)(DIMS, fmac_dims, out_str, out, input_str, input, phi_str, phi);
}

/* Resize operator. */
static void R(long input_dims[DIMS], const complex float* input,
							long sx, long wx, bool adj,
							long out_dims[DIMS],				 complex float* out)
{
	md_copy_dims(DIMS, out_dims, input_dims);
	out_dims[0] = (adj ? sx : wx);
	md_resize_center(DIMS, out_dims, out, input_dims, input, CFL_SIZE);
}

/* Wave operator. */
static void W(long input_dims[DIMS], const complex float* input,
							long wave_dims[DIMS],  const complex float* wave, 
							bool adj,
							long out_dims[DIMS],				 complex float* out)
{
	long input_str[DIMS];
	md_calc_strides(DIMS, input_str, input_dims, CFL_SIZE);

	long wave_str[DIMS];
	md_calc_strides(DIMS, wave_str, wave_dims, CFL_SIZE);

	long fmac_dims[DIMS];
	md_merge_dims(DIMS, fmac_dims, input_dims, wave_dims);
	long fmac_str[DIMS];
	md_calc_strides(DIMS, fmac_str, fmac_dims, CFL_SIZE);

	md_copy_dims(DIMS, out_dims, input_dims);
	long out_str[DIMS];
	md_calc_strides(DIMS, out_str, out_dims, CFL_SIZE);

	(adj ? md_zfmacc2 : md_zfmac2)(DIMS, fmac_dims, out_str, out, input_str, input, wave_str, wave);
}

/* Fourier along readout. */
static void Fx(long input_dims[DIMS], const complex float* input,
							 bool adj,
							 long out_dims[DIMS],					complex float* out)
{
	md_copy_dims(DIMS, out_dims, input_dims);
	(adj ? ifftuc(DIMS, input_dims, 1, out, input) : fftuc(DIMS, input_dims, 1, out, input));
}

/* Fourier along phase encode directions. */
static void Fyz(long input_dims[DIMS], const complex float* input,
							 bool adj,
							 long out_dims[DIMS],					 complex float* out)
{
	md_copy_dims(DIMS, out_dims, input_dims);
	(adj ? ifftuc(DIMS, input_dims, 6, out, input) : fftuc(DIMS, input_dims, 6, out, input));
}

/* Sampling operator. */
static void M(long input_dims[DIMS], complex float* input,
							long mask_dims[DIMS],  complex float* mask, 
							long out_dims[DIMS],	 complex float* out)
{
	long input_str[DIMS];
	md_calc_strides(DIMS, input_str, input_dims, CFL_SIZE);

	long mask_str[DIMS];
	md_calc_strides(DIMS, mask_str, mask_dims, CFL_SIZE);

	long fmac_dims[DIMS];
	md_merge_dims(DIMS, fmac_dims, input_dims, mask_dims);

	md_copy_dims(DIMS, out_dims, input_dims);
	long out_str[DIMS];
	md_calc_strides(DIMS, out_str, out_dims, CFL_SIZE);

	md_zfmac2(DIMS, fmac_dims, out_str, out, input_str, input, mask_str, mask);
}

/* Construction sampling temporal kernel. */
static void construct_kernel(long mask_dims[DIMS], complex float* mask,
														 long phi_dims[DIMS],  complex float* phi, 
														 long kern_dims[DIMS], complex float* kern)
{
	long sy = mask_dims[1];
	long sz = mask_dims[2];
	long tf = phi_dims[5];
	long tk = phi_dims[6];

	long cvec_dims[] = { [0 ... DIMS - 1] = 1 };
	cvec_dims[6] = tk;
	complex float cvec[tk];
	md_clear(DIMS, cvec_dims, cvec, CFL_SIZE);

	long tvec_dims[] = { [0 ... DIMS - 1] = 1 };
	tvec_dims[5] = tf;
	complex float mvec[tf];
	complex float tvec1[tf];
	complex float tvec2[tf];

	long out_dims[] = { [0 ... DIMS - 1] = 1 };
	out_dims[0] = tk;
	out_dims[1] = sy;
	out_dims[2] = sz;
	out_dims[3] = tk;
	complex float* out	= md_calloc(DIMS, out_dims, CFL_SIZE);

	for (int y = 0; y < sy; y ++) {
		for (int z = 0; z < sz; z ++) {

			for (int t = 0; t < tf; t ++)
				mvec[t] = mask[(y + sy * z) + (sy * sz) * t];

			for (int t = 0; t < tk; t ++) {
				cvec[t] = 1;
				md_clear(DIMS, tvec_dims, tvec1, CFL_SIZE);
				P(cvec_dims, cvec,	phi_dims,  phi,  false, tvec_dims, tvec1);
				md_clear(DIMS, tvec_dims, tvec2, CFL_SIZE);
				M(tvec_dims, tvec1, tvec_dims, mvec,				tvec_dims, tvec2);
				P(tvec_dims, tvec2, phi_dims,  phi,  true,	cvec_dims, 
					out + (((0 + y * tk) + z * sy * tk) + t * sy * sz * tk));
				cvec[t] = 0;
			}
		}
	}

	unsigned int permute_order[DIMS] = {4, 1, 2, 5, 6, 7, 3, 0};
	for (unsigned int i = 8; i < DIMS; i++)
		permute_order[i] = i;

	md_permute(DIMS, permute_order, kern_dims, kern, out_dims, out, CFL_SIZE);

	md_free(out);
}

/* Sampling-temporal operator. */
static void K(long input_dims[DIMS],	const complex float* input,
							long kernel_dims[DIMS], const complex float* kernel, 
							long out_dims[DIMS],					complex float* out)
{
	long input_str[DIMS];
	md_calc_strides(DIMS, input_str, input_dims, CFL_SIZE);

	long kernel_str[DIMS];
	md_calc_strides(DIMS, kernel_str, kernel_dims, CFL_SIZE);

	long fmac_dims[DIMS];
	md_merge_dims(DIMS, fmac_dims, input_dims, kernel_dims);

	md_copy_dims(DIMS, out_dims, input_dims);
	out_dims[COEFF2_DIM] = out_dims[COEFF_DIM];
	out_dims[COEFF_DIM] = 1;
	long out_str[DIMS];
	md_calc_strides(DIMS, out_str, out_dims, CFL_SIZE);

	md_zfmac2(DIMS, fmac_dims, out_str, out, input_str, input, kernel_str, kernel);
	out_dims[COEFF_DIM] = out_dims[COEFF2_DIM];
	out_dims[COEFF2_DIM] = 1;
}

/* Convert collapsed data into projection. */
static void construct_projection(long collapsed_dims[DIMS], const complex float* collapsed,
																 long maps_dims[DIMS],			const complex float* maps,
																 long wave_dims[DIMS],			const complex float* wave,
																 long coeff_dims[DIMS],						complex float* out)
{
	long flatten_dims[] = {collapsed_dims[0] * collapsed_dims[1] * collapsed_dims[2] *
												 collapsed_dims[3] * collapsed_dims[6]};
	complex float* buffer1 = md_alloc_sameplace(1, flatten_dims, CFL_SIZE, out);
	complex float* buffer2 = md_alloc_sameplace(1, flatten_dims, CFL_SIZE, out);

	long apply1_dims[] = { [0 ... DIMS - 1] = 1 };
	long apply2_dims[] = { [0 ... DIMS - 1] = 1 };

	md_clear(1, flatten_dims, buffer1, CFL_SIZE);
	Fyz(collapsed_dims, collapsed, true, apply1_dims, buffer1);

	md_clear(1, flatten_dims, buffer2, CFL_SIZE);
	W(apply1_dims, buffer1, wave_dims, wave, true, apply2_dims, buffer2);

	md_clear(1, flatten_dims, buffer1, CFL_SIZE);
	Fx(apply2_dims, buffer2, true, apply1_dims, buffer1);

	md_clear(1, flatten_dims, buffer2, CFL_SIZE);
	R(apply1_dims, buffer1, maps_dims[0], wave_dims[0], true, apply2_dims, buffer2);

	md_clear(1, flatten_dims, buffer1, CFL_SIZE);
	E(apply2_dims, buffer2, maps_dims, maps, true, apply1_dims, buffer1);

	float n = sqrt(md_zscalar(DIMS, apply1_dims, buffer1, buffer1));
	md_zsmul(DIMS, coeff_dims, out, buffer1, 1. / n);

	md_free(buffer1);
	md_free(buffer2);
}

static DEF_TYPEID(wshfl_s);

struct wshfl_s {
	INTERFACE(linop_data_t);

	unsigned int N;
	unsigned int D;

	long* maps_dims;
	long* wave_dims;
	long* kern_dims;

	long* coeff_dims;

	const complex float* maps;
	const complex float* wave;
	const complex float* kern;
};

/* To trick the solver since we're working with the normal operator. */
static void wshfl_adj(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct wshfl_s* data = CAST_DOWN(wshfl_s, _data);
	md_copy(data->N, data->coeff_dims, dst, src, CFL_SIZE);
}

static void wshfl_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct wshfl_s* data = CAST_DOWN(wshfl_s, _data);

	const complex float* maps = data->maps;
	const complex float* wave = data->wave;
	const complex float* kern = data->kern;

	long flatten_dims[] = {data->D};
	complex float* buffer1 = md_alloc_sameplace(1, flatten_dims, CFL_SIZE, src);
	complex float* buffer2 = md_alloc_sameplace(1, flatten_dims, CFL_SIZE, src);

	long apply1_dims[] = { [0 ... DIMS - 1] = 1 };
	long apply2_dims[] = { [0 ... DIMS - 1] = 1 };

	md_clear(1, flatten_dims, buffer1, CFL_SIZE);
	E(data->coeff_dims, src, data->maps_dims,  maps, false, apply1_dims, buffer1);

	md_clear(1, flatten_dims, buffer2, CFL_SIZE);
	R(apply1_dims, buffer1, data->maps_dims[0], data->wave_dims[0], false, apply2_dims, buffer2);

	md_clear(1, flatten_dims, buffer1, CFL_SIZE);
	Fx(apply2_dims, buffer2, false, apply1_dims, buffer1);

	md_clear(1, flatten_dims, buffer2, CFL_SIZE);
	W(apply1_dims, buffer1, data->wave_dims, wave, false, apply2_dims, buffer2);

	md_clear(1, flatten_dims, buffer1, CFL_SIZE);
	Fyz(apply2_dims, buffer2, false, apply1_dims, buffer1);

	md_clear(1, flatten_dims, buffer2, CFL_SIZE);
	K(apply1_dims, buffer1, data->kern_dims, kern, apply2_dims, buffer2);

	md_clear(1, flatten_dims, buffer1, CFL_SIZE);
	Fyz(apply2_dims, buffer2, true, apply1_dims, buffer1);

	md_clear(1, flatten_dims, buffer2, CFL_SIZE);
	W(apply1_dims, buffer1, data->wave_dims, wave, true, apply2_dims, buffer2);

	md_clear(1, flatten_dims, buffer1, CFL_SIZE);
	Fx(apply2_dims, buffer2, true, apply1_dims, buffer1);

	md_clear(1, flatten_dims, buffer2, CFL_SIZE);
	R(apply1_dims, buffer1, data->maps_dims[0], data->wave_dims[0], true, apply2_dims, buffer2);

	E(apply2_dims, buffer2, data->maps_dims, maps, true, data->coeff_dims, dst);

	md_free(buffer1);
	md_free(buffer2);
}

static void wshfl_free(const linop_data_t* _data)
{
	const struct wshfl_s* data = CAST_DOWN(wshfl_s, _data);

	xfree(data->maps_dims);
	xfree(data->wave_dims);
	xfree(data->kern_dims);
	xfree(data->coeff_dims);

	xfree(data);
}

static struct linop_s* linop_wshfl_create(long N, long D, long _coeff_dims[N], 
																					long _maps_dims[N], complex float* maps, 
																					long _wave_dims[N], complex float* wave,
																					long _kern_dims[N], complex float* kern)
{
	PTR_ALLOC(struct wshfl_s, data);
	SET_TYPEID(wshfl_s, data);

	data->N = N;
	data->D = D;

	PTR_ALLOC(long[N], coeff_dims);
	md_copy_dims(N, *coeff_dims, _coeff_dims);
	data->coeff_dims = *PTR_PASS(coeff_dims);

	PTR_ALLOC(long[N], maps_dims);
	PTR_ALLOC(long[N], wave_dims);
	PTR_ALLOC(long[N], kern_dims);

	md_copy_dims(N, *maps_dims, _maps_dims);
	md_copy_dims(N, *wave_dims, _wave_dims);
	md_copy_dims(N, *kern_dims, _kern_dims);

	data->maps_dims = *PTR_PASS(maps_dims);
	data->wave_dims = *PTR_PASS(wave_dims);
	data->kern_dims = *PTR_PASS(kern_dims);

	data->maps = maps;
	data->wave = wave;
	data->kern = kern;

	return linop_create(N, _coeff_dims, N, _coeff_dims, CAST_UP(PTR_PASS(data)), 
		wshfl_normal, wshfl_adj, wshfl_normal, NULL, wshfl_free);
}

int main_wshfl(int argc, char* argv[])
{
	double start_time = timestamp();

	float lambda	= 1E-3;
	int		maxiter = 300;
	int		blksize = 8;
	float step		= 0.25;
	float tol			= 1.E-2;
	bool	llr			= false;
	bool	wav			= false;
	int		gpun		= -1;
	bool	fista		= false;
	float cont		= 1;
							 
	const struct opt_s opts[] = {
		OPT_FLOAT('r', &lambda,  "lambda", "Soft threshold lambda for wavelet or locally low rank."),
		OPT_INT(	'b', &blksize, "blkdim", "Block size for locally low rank."),
		OPT_INT(	'i', &maxiter, "mxiter", "Maximum number of iterations."),
		OPT_FLOAT('s', &step,		 "step",	 "Step size for iterative method."),
		OPT_FLOAT('c', &cont,		 "cntnu",  "Continuation value for IST/FISTA."),
		OPT_FLOAT('t', &tol,		 "tol",		 "Tolerance convergence condition for iterative method."),
		OPT_INT(	'g', &gpun,		 "gpun",	 "Set GPU device number. If not set, use CPU."),
		OPT_SET(	'f', &fista,						 "Reconstruct using FISTA instead of IST."),
		OPT_SET(	'w', &wav,							 "Use wavelet."),
		OPT_SET(	'l', &llr,							 "Use locally low rank."),
	};

	cmdline(&argc, argv, 6, 6, usage_str, help_str, ARRAY_SIZE(opts), opts);

	debug_printf(DP_INFO, "Loading data... ");

	long maps_dims[DIMS];
	complex float* maps = load_cfl(argv[1], DIMS, maps_dims);
		 
	long wave_dims[DIMS];
	complex float* wave = load_cfl(argv[2], DIMS, wave_dims);
					
	long phi_dims[DIMS];
	complex float* phi = load_cfl(argv[3], DIMS, phi_dims);
							 
	long reorder_dims[DIMS];
	complex float* reorder = load_cfl(argv[4], DIMS, reorder_dims);
										
	long data_dims[DIMS];
	complex float* data = load_cfl(argv[5], DIMS, data_dims);

	debug_printf(DP_INFO, "Done.\n");

#ifdef USE_CUDA
	if (gpun != -1) {
		debug_printf(DP_INFO, "Transferring maps and wave to GPU memory... ");
		complex float* tmp = maps;
		maps = md_gpu_move(DIMS, maps_dims, tmp, CFL_SIZE);
		unmap_cfl(DIMS, maps_dims, maps);

		tmp = wave;
		wave = md_gpu_move(DIMS, wave_dims, tmp, CFL_SIZE);
		unmap_cfl(DIMS, wave_dims, wave);
		debug_printf(DP_INFO, "Done.\n");
	}
#endif

	int wx = wave_dims[0];
	int sx = maps_dims[0];
	int sy = maps_dims[1];
	int sz = maps_dims[2];
	int nc = maps_dims[3];
	int md = maps_dims[4];
	int tf = phi_dims[5];
	int tk = phi_dims[6];

	long coeff_dims[] = { [0 ... DIMS - 1] = 1 };
	coeff_dims[0] = sx; 
	coeff_dims[1] = sy; 
	coeff_dims[2] = sz;
	coeff_dims[4] = md; 
	coeff_dims[6] = tk;

#ifdef USE_CUDA
	if (gpun != -1) 
		num_init_gpu_device(gpun);
	else
		num_init();
#else
	num_init();
#endif

	debug_printf(DP_INFO, "Constructing sampling mask from reorder table... ");
	long mask_dims[] = { [0 ... DIMS - 1] = 1 };
	mask_dims[1] = sy;
	mask_dims[2] = sz;
	mask_dims[5] = tf;
	complex float* mask = md_alloc_sameplace(DIMS, mask_dims, CFL_SIZE, maps);
	construct_mask(reorder_dims, reorder, mask_dims, mask);
	debug_printf(DP_INFO, "Done.\n");

	debug_printf(DP_INFO, "Collapsing data table... ");

	long collapse_dims[] = { [0 ... DIMS - 1] = 1 };
	collapse_dims[0] = wx;
	collapse_dims[1] = sy;
	collapse_dims[2] = sz;
	collapse_dims[3] = nc;
	collapse_dims[6] = tk;
	complex float* collapse = md_alloc_sameplace(DIMS, collapse_dims, CFL_SIZE, maps); 
	collapse_table(reorder_dims, reorder, phi_dims, phi, data_dims, data, collapse_dims, collapse);
	unmap_cfl(DIMS, reorder_dims, reorder);
	unmap_cfl(DIMS, data_dims, data);
	debug_printf(DP_INFO, "Done.\n");

	debug_printf(DP_INFO, "Constructing projection... ");
	complex float* proj = md_alloc_sameplace(DIMS, coeff_dims, CFL_SIZE, maps); 
	construct_projection(collapse_dims, collapse, maps_dims, maps, wave_dims, wave, coeff_dims, proj);
	md_free(collapse);
	debug_printf(DP_INFO, "Done.\n");

	debug_printf(DP_INFO, "Constructing sampling-temporal kernel... ");
	long kern_dims[] = { [0 ... DIMS - 1] = 1 };
	kern_dims[1] = sy;
	kern_dims[2] = sz;
	kern_dims[6] = tk;
	kern_dims[7] = tk;
	complex float* kern = md_alloc_sameplace(DIMS, kern_dims, CFL_SIZE, maps); 
	construct_kernel(mask_dims, mask, phi_dims, phi, kern_dims, kern);
	unmap_cfl(DIMS, phi_dims, phi);
	md_free(mask);
	debug_printf(DP_INFO, "Done.\n");

	debug_printf(DP_INFO, "Creating linear operator:\n");
	const struct linop_s* wshfl_op = linop_wshfl_create(DIMS, wx * sy * sz * nc * tk, coeff_dims, 
																											maps_dims, maps,
																											wave_dims, wave,
																											kern_dims, kern);
	print_opdims(wshfl_op);
	debug_printf(DP_INFO, "Done.\n");

	const struct operator_p_s* threshold_op = NULL;
	long blkdims[MAX_LEV][DIMS];
	long minsize[] = { [0 ... DIMS - 1] = 1 };
	minsize[0] = MIN(sx, 16);
	minsize[1] = MIN(sy, 16);
	minsize[2] = MIN(sz, 16);
	unsigned int WAVFLAG = (sx > 1) * READ_FLAG | (sy > 1) * PHS1_FLAG | (sz > 2) * PHS2_FLAG;

	if ((wav == true) || (llr == true)) {
		if (wav) {
			debug_printf(DP_INFO, "Creating wavelet threshold operator... ");
			threshold_op = prox_wavelet_thresh_create(DIMS, coeff_dims, WAVFLAG, 0u, minsize, lambda, false);
		} else {
			debug_printf(DP_INFO, "Creating locally low rank threshold operator... ");
			llr_blkdims(blkdims, ~COEFF_DIM, coeff_dims, blksize);
			threshold_op = lrthresh_create(coeff_dims, true, ~COEFF_FLAG, (const long (*)[])blkdims, 
				lambda, false, false);
		}
		debug_printf(DP_INFO, "Done.\n");
	}

	italgo_fun_t italgo = NULL;
	iter_conf*	 iconf	= NULL;

	struct iter_conjgrad_conf cgconf;
	struct iter_fista_conf		fsconf;
	struct iter_ist_conf			isconf;

	if ((wav == false) && (llr == false)) {
		cgconf							= iter_conjgrad_defaults;
		cgconf.maxiter			= maxiter;
		cgconf.l2lambda			= 0;
		cgconf.tol					= tol;
		italgo							= iter_conjgrad;
		iconf								= CAST_UP(&cgconf);
		debug_printf(DP_INFO, "Using conjugate gradient.\n");
	} else if (fista) {
		fsconf							= iter_fista_defaults;
		fsconf.maxiter			= maxiter;
		fsconf.step					= step;
		fsconf.hogwild			= false;
		fsconf.tol					= tol;
		fsconf.continuation = cont;
		italgo							= iter_fista;
		iconf								= CAST_UP(&fsconf);
		debug_printf(DP_INFO, "Using FISTA.\n");
	} else {
		isconf							= iter_ist_defaults;
		isconf.step					= step;
		isconf.maxiter			= maxiter;
		isconf.tol					= tol;
		isconf.continuation = cont;
		italgo							= iter_ist;
		iconf								= CAST_UP(&isconf);
		debug_printf(DP_INFO, "Using IST.\n");
	}

	debug_printf(DP_INFO, "Starting reconstruction... ");
	complex float* recon = md_alloc_sameplace(DIMS, coeff_dims, CFL_SIZE, maps); 
	struct lsqr_conf lsqr_conf = { 0., gpun != -1 };
	lsqr(DIMS, &lsqr_conf, italgo, iconf, wshfl_op, threshold_op, coeff_dims, recon,
		coeff_dims, proj, NULL);
	debug_printf(DP_INFO, "Done.\n");

	debug_printf(DP_INFO, "Cleaning up and saving result... ");
	linop_free(wshfl_op);
	md_free(kern);
#ifdef USE_CUDA
	if (gpun != -1) {
		md_free(maps);
		md_free(wave);
	} else {
		unmap_cfl(DIMS, maps_dims, maps);
		unmap_cfl(DIMS, wave_dims, wave);
	}
#else
	unmap_cfl(DIMS, maps_dims, maps);
	unmap_cfl(DIMS, wave_dims, wave);
#endif

	complex float* result = create_cfl(argv[6], DIMS, coeff_dims);
	md_copy(DIMS, coeff_dims, result, recon, CFL_SIZE);
	unmap_cfl(DIMS, coeff_dims, result);
	md_free(recon);
	debug_printf(DP_INFO, "Done.\n");

	double end_time = timestamp();
	debug_printf(DP_INFO, "Total Time: %f seconds.\n", end_time - start_time);

	return 0;
}
