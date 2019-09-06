/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Siddharth Iyer <sid8795@gmail.com>
 * 2015-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/init.h"

#include "misc/debug.h"

#include "estnoise/estnoise.h"


static const char usage_str[] = "<kspace>";
static const char help_str[] = "Estimate the noise variance assuming white Gaussian noise.";


int main_estvar(int argc, char* argv[])
{
	bool wavelet = true;
	const struct opt_s opts[] = {
		OPT_SET(   'w', &wavelet,  "Estimate noise from wavelet projection."),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;
	long ksp_dims[N];

	complex float* ksp = load_cfl(argv[1], N, ksp_dims);

	float variance = estvar_ksp(N, ksp_dims, ksp);

	unmap_cfl(N, ksp_dims, ksp);

	bart_printf("Estimated noise variance: %f\n", variance);

	return 0;
}
