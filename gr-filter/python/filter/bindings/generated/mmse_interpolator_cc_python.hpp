

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_FILTER_MMSE_INTERPOLATOR_CC_PYTHON_HPP
#define INCLUDED_GR_FILTER_MMSE_INTERPOLATOR_CC_PYTHON_HPP

#include <gnuradio/filter/mmse_interpolator_cc.h>

void bind_mmse_interpolator_cc(py::module& m)
{
    using mmse_interpolator_cc    = gr::filter::mmse_interpolator_cc;


    py::class_<mmse_interpolator_cc,gr::block,
        std::shared_ptr<mmse_interpolator_cc>>(m, "mmse_interpolator_cc")

        .def(py::init(&mmse_interpolator_cc::make),
           py::arg("phase_shift"), 
           py::arg("interp_ratio") 
        )
        

        .def("mu",&mmse_interpolator_cc::mu)
        .def("interp_ratio",&mmse_interpolator_cc::interp_ratio)
        .def("set_mu",&mmse_interpolator_cc::set_mu,
            py::arg("mu") 
        )
        .def("set_interp_ratio",&mmse_interpolator_cc::set_interp_ratio,
            py::arg("interp_ratio") 
        )
        .def("to_basic_block",[](std::shared_ptr<mmse_interpolator_cc> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_FILTER_MMSE_INTERPOLATOR_CC_PYTHON_HPP */
