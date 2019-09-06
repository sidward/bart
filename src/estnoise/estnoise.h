/* Copyright 2019. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2019 Siddharth Iyer <ssi@mit.edu>
 */

#ifndef __ESTNOISE_H
#define __ESTNOISE_H
 
#include "misc/cppwrap.h"
#include "misc/mri.h"

extern float estvar_ksp(long N, const long ksp_dims[N], const complex float* ksp);

#include "misc/cppwrap.h"
#endif	// __ESTNOISE_H
