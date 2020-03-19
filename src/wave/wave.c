/* Copyright 2020. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2020 Siddharth Iyer <ssi@mit.edu>
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "wave/wave.h"

// Larmor frequency in Hertz per Gauss
#ifndef LARMOR
#define LARMOR 4257.56
#endif

/*Generate wave phase per centimeter.
  Args:
    wx:      Size of readout after wave oversampling.
    adc:     Readout duration in microseconds.
    dt:      ADC sampling rate in seconds. Typically: 1E-5.
    cyc:     Number of sinusoidal cycle.
    gm:      Gradient maximum amplitude in Gauss/cm.
    sm:      Maximum gradient slew rate in Gauss/cm/s.
    issine:  If true, contrust sine wave. Else, construct cosine.
    corr_dt: Gradient delay correction factor.
    corr_sc: Gradient scaling correction factor.
    out:     Save result to out. Must be 1D array of size wx.*/
extern void phase_per_cm(long wx, long adc, float dt, long cyc, float gm, float sm, _Bool issine, float corr_dt, float corr_sc, complex float* out)
{
	assert(0 == adc % 10);					// Scanners require ADC_duration to be a multiple of 10.
	int wavepoints = adc/10;				// Number of points in the gradient wave.
	float T = wavepoints * dt/cyc;  // Time period of the sine wave.
	float w = 2 * M_PI/T;						// Frequency in radians per second.

	/* Calculating the wave-amplitude to use. It is either slew limited or gradient amplitude limited. */
	float gamp = (sm >= w * gm) ? gm : sm/w;
	float gwave[wavepoints];
	for (int tdx = 0; tdx < wavepoints; tdx++)
		gwave[tdx] = (corr_sc * gamp) * ((issine) ? sin(w * (tdx * dt - corr_dt)) : cos(w * (tdx * dt - corr_dt)));
	
	complex float phasepercm[wavepoints];
	float prephase = -2 * M_PI * LARMOR * gamp/w;
	float cumsum = 0;
	for (int tdx = 0; tdx < wavepoints; tdx++) {
		phasepercm[tdx] = 2 * M_PI * LARMOR * (cumsum + gwave[tdx]/2.0) * dt + prephase;
		cumsum = cumsum + gwave[tdx]; 
	}

	// Interpolate to wx via sinc interpolation
	const long wavepoint_dims[1] = {wavepoints};
	const long interp_dims[1] = {wx};

	complex float k_phasepercm[wavepoints]; 
	fftuc(1, wavepoint_dims, 1, k_phasepercm, phasepercm);	

	complex float k_phasepercm_interp[wx]; 
	md_resize_center(1, interp_dims, k_phasepercm_interp, wavepoint_dims, k_phasepercm, 
		sizeof(complex float));

	complex float phasepercm_interp_complex[wx]; 
	ifftuc(1, interp_dims, 1, phasepercm_interp_complex, k_phasepercm_interp);

	complex float phasepercm_interp_real[wx]; 
	md_zreal(1, interp_dims, phasepercm_interp_real, phasepercm_interp_complex);

	float scale = sqrt((float) wx/wavepoints);
	md_zsmul(1, interp_dims, out, phasepercm_interp_real, scale);
}

extern void psf_from_phasepercm(long wx, long sy, float dy, float offset, complex float* phasepercm, complex float* psf)
{
  complex float* tmp[sy][wx];
	int midy = sy/2;
	complex float phase[wx];
	float val;
	const long dims[1] = {wx};

	for (int ydx = 0; ydx < sy; ydx++) {
		val = dy * (midy - ydx) - offset;
		md_zsmul(1, dims, phase, phasepercm, val);
		md_zexpj(1, dims, tmp[ydx], phase);
	}
  const long psf_dims[2] = {wx, sy};
  md_copy(2, psf_dims, psf, tmp, sizeof(complex float));
}
