

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_FILTER_SINGLE_POLE_IIR_FILTER_CC_PYTHON_HPP
#define INCLUDED_GR_FILTER_SINGLE_POLE_IIR_FILTER_CC_PYTHON_HPP

#include <gnuradio/filter/single_pole_iir_filter_cc.h>

void bind_single_pole_iir_filter_cc(py::module& m)
{
    using single_pole_iir_filter_cc    = gr::filter::single_pole_iir_filter_cc;


    py::class_<single_pole_iir_filter_cc,gr::sync_block,
        std::shared_ptr<single_pole_iir_filter_cc>>(m, "single_pole_iir_filter_cc")

        .def(py::init(&single_pole_iir_filter_cc::make),
           py::arg("alpha"), 
           py::arg("vlen") = 1 
        )
        

        .def("set_taps",&single_pole_iir_filter_cc::set_taps,
            py::arg("alpha") 
        )
        .def("to_basic_block",[](std::shared_ptr<single_pole_iir_filter_cc> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_FILTER_SINGLE_POLE_IIR_FILTER_CC_PYTHON_HPP */
