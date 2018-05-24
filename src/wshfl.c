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

/* Helper function to print out operator dimensions. */
static void print_opdims(const struct linop_s* op) 
{
  const struct iovec_s* domain = linop_domain(op);
  const struct iovec_s* codomain = linop_codomain(op);
  debug_printf(DP_INFO, "domain:");
  for (unsigned int idx = 0; idx < domain->N; idx ++)
    debug_printf(DP_INFO, " %ld", domain->dims[idx]);
  debug_printf(DP_INFO, "\n");
  debug_printf(DP_INFO, "codomain:");
  for (unsigned int idx = 0; idx < codomain->N; idx ++)
    debug_printf(DP_INFO, " %ld", codomain->dims[idx]);
  debug_printf(DP_INFO, "\n");
}

/* Construct sampling mask from reorder tables. */
static void construct_mask(long reorder_dims[WDIM], complex float* reorder, 
                           long mask_dims[WDIM], complex float* mask)
{
  int n = reorder_dims[0];
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

/* Collapse over the time dimension in a memory efficient manner. */
static void  collapse_data(long reorder_dims[WDIM], complex float* reorder, 
                           const struct linop_s* phi_op, 
                           long data_dims[WDIM], complex float* data, 
                           long collapse_dims[WDIM], complex float* collapse)
{
  long wx = collapse_dims[0];
  long sy = collapse_dims[1];
  long sz = collapse_dims[2];
  long nc = collapse_dims[3]; 
  long md = collapse_dims[4]; // Should be 1.
  long tf = phi_dims[5];
  long tk = phi_dims[6];
  long n = reorder_dims[0];

  long vec_dims = {wx, 1, 1, nc, 1, tf, 1};
  long vec_copy_dims = {wx, nc, 1, 1, 1, 1, 1};
  long vec_out_dims = {wx, 1, 1, nc, 1, 1, tk};
  complex float vec[tf][nc][sx];

  int ky = -1;
  int kz = -1;
  int t = -1;

  // Sweep through all (ky, kz) points.
  for (int ydx = 0; ydx < sy; ydx ++) {
    for (int zdx = 0; zdx < sz; zdx ++) {

      md_clear(WDIM, vec_dims, vec, sizeof(complex float));

      // Search reorder table for matching indexs.
      for (int ndx = 0; ndx < n; ndx ++) {

        ky = reorder[ndx];
        kz = reorder[ndx + n];
        t  = reorder[ndx + 2 * n];

        if ((ky == ydx) && (kz == zdx)) {
	        md_copy(WDIM, vec_copy_dims, vec[t], data[ndx], sizeof(complex float));
        }

      }

      linop_adjoint(phi_op, WDIM, vec_out_dims, collapse[tdx], //TODO
			unsigned int SN, const long sdims[SN], const complex float* src)
    }
  }
}

int main_wshfl(int argc, char* argv[])
{
  float lambda = 1E-6;
  int maxiter = 50;
  int blksize = 8;
  float step = 0.95;
  float tol = 1.E-3;
  bool llr = false;
           
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

  long coeff_dims[] = {sx, sy, sz, 1, md, 1, tk};

  long mask_dims[] = {1, sy, sz, 1, 1, tf, 1};
  complex float* mask = md_alloc(WDIM, mask_dims, sizeof(complex float));
  construct_mask(reorder_dims, reorder, mask_dims, mask);

  long maps_op_dims[] = {sx, sy, sz, nc, md, 1, tk};
  const struct linop_s* maps_op = linop_fmac_create(WDIM, maps_op_dims, MAPS_FLAG, 
    COIL_FLAG, ~(FFT_FLAGS|COIL_FLAG|MAPS_FLAG), maps);
  debug_printf(DP_INFO, "ESPIRiT operator information.\n");
  print_opdims(maps_op);

  long resize_op_in_dims[]  = {sx, sy, sz, nc, 1, 1, tk};
  long resize_op_out_dims[] = {wx, sy, sz, nc, 1, 1, tk};
  const struct linop_s* resize_op = linop_resize_create(WDIM, resize_op_out_dims, resize_op_in_dims);
  debug_printf(DP_INFO, "Resize operator information.\n");
  print_opdims(resize_op);

  long fft_op_dims[] = {wx, sy, sz, nc, 1, 1, tk};
  const struct linop_s* fx_op = linop_fftc_create(WDIM, fft_op_dims, READ_FLAG);
  const struct linop_s* wave_op = linop_cdiag_create(WDIM, fft_op_dims, FFT_FLAGS, wave);
  const struct linop_s* fyz_op = linop_fftc_create(WDIM, fft_op_dims, PHS1_FLAG|PHS2_FLAG);
  debug_printf(DP_INFO, "Fx operator information.\n");
  print_opdims(fx_op);
  debug_printf(DP_INFO, "Wave operator information.\n");
  print_opdims(wave_op);
  debug_printf(DP_INFO, "Fyz operator information.\n");
  print_opdims(fyz_op);

  long phi_op_dims[] = {wx, sy, sz, nc, 1, tf, tk};
  const struct linop_s* phi_op = linop_fmac_create(WDIM, phi_op_dims, COEFF_FLAG, 
      TE_FLAG, ~(TE_FLAG|COEFF_FLAG), phi);
  debug_printf(DP_INFO, "Phi operator information.\n");
  print_opdims(phi_op);

  long mask_op_dims[] = {wx, sy, sz, nc, 1, tf, tk};
  const struct linop_s* mask_op = linop_cdiag_create(WDIM, mask_op_dims, PHS1_FLAG|PHS2_FLAG|TE_FLAG, mask);
  debug_printf(DP_INFO, "Mask operator information.\n");
  print_opdims(mask_op);

  long blkdims[MAX_LEV][DIMS];
  llr_blkdims(blkdims, ~COEFF_DIM, coeff_dims, blksize);
  long minsize[] = {MIN(sx, 32), MIN(sy, 32), MIN(sz, 32), 1, 1, 1, 1};
  const struct operator_p_s* threshold_op = ((llr) ?
    (lrthresh_create(coeff_dims, true, ~COEFF_DIM, (const long (*)[])blkdims, lambda, false, false)) :
    (prox_wavelet_thresh_create(WDIM, coeff_dims, FFT_FLAGS, 0u, minsize, lambda, false)));

  complex float* proj = md_alloc(WDIM, coeff_dims, sizeof(complex float));
  construct_proj(reorder_dims, reorder, phi_dims, phi, data_dims, data);


  linop_free(maps_op);
  linop_free(resize_op);
  linop_free(fx_op);
  linop_free(wave_op);
  linop_free(fyz_op);
  linop_free(phi_op);
  linop_free(mask_op);

  unmap_cfl(WDIM, maps_dims,    maps);
  unmap_cfl(WDIM, wave_dims,    wave);
  unmap_cfl(WDIM, phi_dims,     phi);
  unmap_cfl(WDIM, reorder_dims, reorder);
  unmap_cfl(WDIM, data_dims,    data);

  return 0;
}
