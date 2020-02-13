

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_INTERPOLATING_RESAMPLER_TYPE_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_INTERPOLATING_RESAMPLER_TYPE_PYTHON_HPP

#include <gnuradio/digital/interpolating_resampler_type.h>

void bind_interpolating_resampler_type(py::module& m)
{

    py::enum_<gr::digital::ir_type>(m,"ir_type")
        .value("IR_NONE", gr::digital::IR_NONE) // -1
        .value("IR_MMSE_8TAP", gr::digital::IR_MMSE_8TAP) // 0
        .value("IR_PFB_NO_MF", gr::digital::IR_PFB_NO_MF) // 1
        .value("IR_PFB_MF", gr::digital::IR_PFB_MF) // 2
        .export_values()
    ;

} 

#endif /* INCLUDED_GR_DIGITAL_INTERPOLATING_RESAMPLER_TYPE_PYTHON_HPP */
