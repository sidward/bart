/* Copyright 2020. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2020 Siddharth Iyer <ssi@mit.edu>
 */

#include "misc/cppwrap.h"

extern void phase_per_cm(long wx, long adc, float dt, long cyc, float gm, float sm, _Bool issine, float corr_dt, float corr_sc, complex float* out);
extern void psf_from_phasepercm(long wx, long sy, float dy, float offset, complex float* phasepercm, complex float* psf);

#include "misc/cppwrap.h"
