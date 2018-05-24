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

const int WDIM = 7;

static const char usage_str[] = "<maps> <wave> <phi> <reorder> <data> <output>";
static const char help_str[] = "Perform wave-shuffling reconstruction.\n\n"
                               "Conventions:\n"
                               "  * (sx, sy, sz) - Spatial dimensions.\n"
                               "  * wx           - Extended FOV in READ_DIM due to\n"
                               "                   wave's voxel spreading.\n"
                               "  * (nc, md)     - Number of channels and ESPIRiT's \n"
                               "                   extended-SENSE model operator\n"
                               "                   dimensions (or # of maps).\n"
                               "  * (tf, tk)     - Turbo-factor and the rank\n"
                               "                   of the temporal basis used in\n"
                               "                   shuffling.\n"
                               "  * ntr          - Number of TRs, or the number of\n"
                               "                   (ky, kz) points acquired of one\n"
                               "                   echo image.\n"
                               "  * n            - Total number of (ky, kz) points\n"
                               "                   acquired. This is equal to the\n"
                               "                   product of ntr and tf.\n\n"
                               "Descriptions:\n"
                               "  * reorder is an (n by 3) index matrix such that\n"
                               "    [ky, kz, t] = reorder(i, :) represents the\n"
                               "    (ky, kz) kspace position of the readout line\n" 
                               "    acquired at echo number (t).\n"
                               "  * data is a (wx by nc by n) matrix such that\n"
                               "    data(:, :, k) represents the kth multichannel\n"
                               "    kspace line.\n\n"
                               "Expected dimensions:\n"
                               "  * maps    - (   sx, sy, sz, nc, md,  1,  1)\n"
                               "  * wave    - (   wx, sy, sz,  1,  1,  1,  1)\n"
                               "  * phi     - (    1,  1,  1,  1,  1, tf, tk)\n"
                               "  * output  - (   sx, sy, sz,  1, md,  1, tk)\n"
															 "  * reorder - (    n,  3,  1,  1,  1,  1,  1)\n"
															 "  * data    - (   wx, nc,  n,  1,  1,  1,  1)";

/* Construct sampling mask from reorder tables. */
static void construct_mask(long reorder_dims[WDIM], complex float* reorder, 
                           long mask_dims[WDIM],    complex float* mask)
{
  int n  = reorder_dims[0];
  int sy = mask_dims[1];
  int sz = mask_dims[2];
  int tf = mask_dims[5];

  int ydx = -1;
  int zdx = -1;
  int tdx = -1;

	complex float m[tf][sz][sy];
  for (int idx = 0; idx < n; idx ++) {
    ydx = reorder[idx];
    zdx = reorder[idx + n];
    tdx = reorder[idx + 2 * n];
    m[tdx][zdx][ydx] = 1;
  }

	md_copy(WDIM, mask_dims, mask, m, sizeof(complex float));
}

/* Collapse table into the temporal basis. */
static void collapse_table(long reorder_dims[WDIM],  complex float* reorder,
                           long phi_dims[WDIM],      complex float* phi, 
                           long data_dims[WDIM],     complex float* data,
                           long collapse_dims[WDIM], complex float* collapse)
{
  long wx = data_dims[0];
  long sy = collapse_dims[1];
  long sz = collapse_dims[2];
  long nc = data_dims[1];
  long n  = data_dims[2];
  long tf = phi_dims[5];
  long tk = phi_dims[6];
  int  t  = -1;

  long vec_dims[]     = {wx * nc,      tf,  1};
  long phi_mat_dims[] = {      1,      tf, tk};
  long phi_out_dims[] = {wx * nc,       1, tk};
  long fmac_dims[]    = {wx * nc,      tf, tk};

  long out_dims[]     = {wx, nc, tk, sy, sz, 1, 1};
  long copy_dim[]     = {wx * nc};

  complex float* vec = md_alloc(   3, vec_dims, CFL_SIZE);
  complex float* out = md_alloc(WDIM, out_dims, CFL_SIZE);

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

  unsigned int permute_order[] = {0, 3, 4, 1, 5, 6, 2};
  long input_dims[] = {wx, nc, tk, sy, sz, 1, 1};

  md_permute(WDIM, permute_order, collapse_dims, collapse, input_dims, out, CFL_SIZE);

  md_free(vec);
  md_free(out);
}

int main_wshfl(int argc, char* argv[])
{
  float lambda  = 1E-6;
  int   maxiter = 50;
  int   blksize = 8;
  float step    = 0.95;
  float tol     = 1.E-3;
  bool  llr     = false;
           
  const struct opt_s opts[] = {
    OPT_FLOAT('r', &lambda,  "lambda", "Soft threshold lambda for wavelet or locally low rank."),
    OPT_INT(  'b', &blksize, "blkdim", "Block size for locally low rank."),
    OPT_INT(  'i', &maxiter, "mxiter", "Maximum number of iterations."),
    OPT_FLOAT('s', &step,    "step",   "Step size for iterative method."),
    OPT_FLOAT('t', &tol,     "tol",    "Tolerance convergence condition for iterative method."),
    OPT_SET(  'l', &llr,               "Use locally low rank instead of wavelet."),
  };

  cmdline(&argc, argv, 6, 6, usage_str, help_str, ARRAY_SIZE(opts), opts);

  long maps_dims[WDIM];
  complex float* maps = load_cfl(argv[1], WDIM, maps_dims);
     
  long wave_dims[WDIM];
  complex float* wave = load_cfl(argv[2], WDIM, wave_dims);
          
  long phi_dims[WDIM];
  complex float* phi = load_cfl(argv[3], WDIM, phi_dims);
               
  long reorder_dims[WDIM];
  complex float* reorder = load_cfl(argv[4], WDIM, reorder_dims);
                    
  long data_dims[WDIM];
  complex float* data = load_cfl(argv[5], WDIM, data_dims);

  debug_printf(DP_INFO, "Map dimensions:\n");
  debug_print_dims(DP_INFO, WDIM, maps_dims);
  debug_printf(DP_INFO, "Wave dimensions:\n");
  debug_print_dims(DP_INFO, WDIM, wave_dims);
  debug_printf(DP_INFO, "Phi dimensions:\n");
  debug_print_dims(DP_INFO, WDIM, phi_dims);
  debug_printf(DP_INFO, "Reorder dimensions:\n");
  debug_print_dims(DP_INFO, WDIM, reorder_dims);
  debug_printf(DP_INFO, "Data dimensions:\n");
  debug_print_dims(DP_INFO, WDIM, data_dims);

  int wx = wave_dims[0];
  int sx = maps_dims[0];
  int sy = maps_dims[1];
  int sz = maps_dims[2];
  int nc = maps_dims[3];
  int md = maps_dims[4];
  int tf = phi_dims[5];
  int tk = phi_dims[6];

  long mask_dims[] = {1, sy, sz, 1, 1, tf, 1};
  complex float* mask = md_alloc(WDIM, mask_dims, sizeof(complex float));
  construct_mask(reorder_dims, reorder, mask_dims, mask);

  long collapse_dims[] = {wx, sy, sz, nc, 1, 1, tk};

	complex float* collapse = create_cfl(argv[6], WDIM, collapse_dims);// TODO change "create cfl"
  collapse_table(reorder_dims, reorder, phi_dims, phi, data_dims, data, collapse_dims, collapse);

  long coeff_dims[] = {sx, sy, sz, 1, md, 1, tk};

  long blkdims[MAX_LEV][DIMS];
  llr_blkdims(blkdims, ~COEFF_DIM, coeff_dims, blksize);
  long minsize[] = {MIN(sx, 32), MIN(sy, 32), MIN(sz, 32), 1, 1, 1, 1};
  const struct operator_p_s* threshold_op = ((llr) ?
    (lrthresh_create(coeff_dims, true, ~COEFF_DIM, (const long (*)[])blkdims, lambda, false, false)) :
    (prox_wavelet_thresh_create(WDIM, coeff_dims, FFT_FLAGS, 0u, minsize, lambda, false)));

  unmap_cfl(WDIM, maps_dims,    maps);
  unmap_cfl(WDIM, wave_dims,    wave);
  unmap_cfl(WDIM, phi_dims,     phi);
  unmap_cfl(WDIM, reorder_dims, reorder);
  unmap_cfl(WDIM, data_dims,    data);

  return 0;
}
