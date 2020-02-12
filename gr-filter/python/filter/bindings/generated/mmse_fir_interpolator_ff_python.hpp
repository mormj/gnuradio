

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_FILTER_MMSE_FIR_INTERPOLATOR_FF_PYTHON_HPP
#define INCLUDED_GR_FILTER_MMSE_FIR_INTERPOLATOR_FF_PYTHON_HPP

#include <gnuradio/filter/mmse_fir_interpolator_ff.h>

void bind_mmse_fir_interpolator_ff(py::module& m)
{
    using mmse_fir_interpolator_ff    = gr::filter::mmse_fir_interpolator_ff;


    py::class_<mmse_fir_interpolator_ff,
        std::shared_ptr<mmse_fir_interpolator_ff>>(m, "mmse_fir_interpolator_ff")

        .def(py::init<>())
        .def(py::init<gr::filter::mmse_fir_interpolator_ff const &>(),           py::arg("arg0") 
        )

        .def("ntaps",&mmse_fir_interpolator_ff::ntaps)
        .def("nsteps",&mmse_fir_interpolator_ff::nsteps)
        .def("interpolate",&mmse_fir_interpolator_ff::interpolate,
            py::arg("input"), 
            py::arg("mu") 
        )
        ;


} 

#endif /* INCLUDED_GR_FILTER_MMSE_FIR_INTERPOLATOR_FF_PYTHON_HPP */
