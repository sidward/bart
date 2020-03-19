/* Copyright 2017-2020. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2020 Siddharth Iyer <ssi@mit.edu>
 *
 * Bilgic B, Gagoski BA, Cauley SF, Fan AP, Polimeni JR, Grant PE, Wald LL, Setsompop K. 
 * Wave‐CAIPI for highly accelerated 3D imaging. Magnetic resonance in medicine. 
 * 2015 Jun 1;73(6):2152-62.
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "num/init.h"
#include "wave/wave.h"

#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

static const char usage_str[] = "<output>";
static const char help_str[] = "Generate a wave PSF in hybrid space.\n"
															 "- Assumes the first dimension is the readout dimension.\n"
															 "- Only generates a 2 dimensional PSF.\n"
															 "- Use reshape and fmac to generate a 3D PSF.\n\n"
															 "3D PSF Example:\n"
															 "bart wavepsf		-x 768 -y 128 -r 0.1 -a 3000 -t 0.00001 -g 0.8 -s 17000 -n 6 wY\n"
															 "bart wavepsf -c -x 768 -y 128 -r 0.1 -a 3000 -t 0.00001 -g 0.8 -s 17000 -n 6 wZ\n"
															 "bart reshape 7 wZ 768 1 128 wZ wZ\n"
															 "bart fmac wY wZ wYZ";

int main_wavepsf(int argc, char* argv[])
{
	
	// Spatial dimensions.
	int wx = 512;				// Number of readout points.
	int sy = 128;				// Number of phase encode points.
	float dy = 0.1;			// Resolution in the phase encode direction in cm.

	// ADC parameters.
	int adc = 3000;			// Readout duration in microseconds.
	float dt = 1e-5;		// ADC sampling rate in seconds.

	// Gradient parameters.
	float gmax = 0.8;		// Maximum gradient amplitude in Gauss per centimeter.
	float smax = 17000; // Maximum slew rate in Gauss per centimeter per second.

	// Wave parameters.
	int ncyc = 6;				// Number of gradient sine-cycles.

	// Sine wave or cosine wave.
	bool sn = false;    // Set to false to use a cosine gradient wave/

	const struct opt_s opts[] = {
		OPT_SET(	'S', &sn,   "Set to use a sine gradient wave"),
		OPT_INT(	'x', &wx,		"RO_dim", "Number of readout points"),
		OPT_INT(	'y', &sy,		"PE_dim", "Number of phase encode points"),
		OPT_FLOAT('r', &dy,		"PE_res", "Resolution of phase encode in cm"),
		OPT_INT(	'a', &adc,	"ADC_T",	"Readout duration in microseconds."),
		OPT_FLOAT('t', &dt,		"ADC_dt", "ADC sampling rate in seconds"),
		OPT_FLOAT('g', &gmax, "gMax",		"Maximum gradient amplitude in Gauss/cm"),
		OPT_FLOAT('s', &smax, "sMax",		"Maximum gradient slew rate in Gauss/cm/second"),
		OPT_INT(	'n', &ncyc, "ncyc",		"Number of cycles in the gradient wave"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	assert(0 == adc % 10); // Scanners require ADC_duration to be a multiple of 10.

	complex float phasepercm[wx]; 
	phase_per_cm(wx, adc, dt, ncyc, gmax, smax, sn, 0, 1, phasepercm);

	const long psf_dims[3] = {wx, sy, 1};
	complex float* psf = create_cfl(argv[1], 3, psf_dims);
	psf_from_phasepercm(wx, sy, dy, 0, phasepercm, psf);
	unmap_cfl(3, psf_dims, psf);

	return 0;
}
