/* Copyright 2019. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2019 Siddharth Iyer <ssi@mit.edu>
 */

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/fft.h"
#include "num/flpmath.h"
#include "num/linalg.h"
#include "num/lapack.h"
#include "num/casorati.h"
#include "num/rand.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/resize.h"
#include "misc/debug.h"
#include "misc/utils.h"
#include "misc/mmio.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/waveop.h"
#include "linops/finite_diff.h"
#include "linops/decompose_complex.h"

#include "estnoise.h"

static int compare_real(const void* data, int a, int b)
{
	const complex float* arr = data;
  float f = crealf(arr[a]) - crealf(arr[b]);
  if (f > 0)
		return 1;
  if (f < 0)
		return -1; 
	return 0;
}

float estvar_ksp(long N, const long ksp_dims[N], const complex float* ksp) {

	long sx = ksp_dims[0];
	long sy = ksp_dims[1];
	long sz = ksp_dims[2];

	long minsize[N];
	md_set_dims(N, minsize, 1);

	minsize[0] = MIN(sx, 4);
	minsize[1] = MIN(sy, 4);
	minsize[2] = MIN(sz, 4);

	unsigned int WAVFLAG = (sx > 1) * READ_FLAG | (sy > 1) * PHS1_FLAG | (sz > 2) * PHS2_FLAG;

	long ksp_str[N];
	md_set_dims(N, ksp_str, 1);
	md_calc_strides(N, ksp_str, ksp_dims, CFL_SIZE);

	const struct linop_s* fourier  = linop_ifftc_create(N, ksp_dims, FFT_FLAGS);
	const struct linop_s* sparse   = linop_wavelet_create(N, WAVFLAG, ksp_dims, ksp_str, minsize, true);
	//const struct linop_s* sparse   = linop_zfinitediff_create(N, ksp_dims, 0, true);
	const struct linop_s* sprfr    = linop_chain_FF(fourier, sparse);
	const struct linop_s* dcx      = linop_decompose_complex_create(N, N - 1, true, linop_codomain(sprfr)->dims);
	const struct linop_s* combine  = linop_chain_FF(sprfr, dcx);
  const struct iovec_s* codomain = linop_codomain(combine);

	complex float* spr = md_alloc_sameplace(N, codomain->dims, CFL_SIZE, ksp);
	md_clear(N, codomain->dims, spr, CFL_SIZE);
	operator_apply(combine->forward, N, codomain->dims, spr, N, ksp_dims, ksp);

  long M = md_calc_size(N, codomain->dims);
  long make_real_dims[1] = { M/2 };
  md_zsmul(1, make_real_dims, spr + M/2, spr + M/2, -1.i);

  long ord_dims[1] = { M };
  int* ord = md_alloc(1, ord_dims, sizeof(int));

  for (int k = 0; k < M; k++)
    ord[k] = k;

	quicksort(M, ord, spr, compare_real);

  float median = crealf(spr[ord[M/2]]);

  md_zsadd(N, codomain->dims, spr, spr, -median);
  md_zabs(N, codomain->dims, spr, spr);

  for (int k = 0; k < M; k++)
    ord[k] = k;

	quicksort(M, ord, spr, compare_real);
  float mad = crealf(spr[ord[M/2]]);

  float stdev = 1.4826 * mad;

	md_free(ord);
	md_free(spr);
  linop_free(combine);

	return 2.0 * powf(stdev, 2);
}
