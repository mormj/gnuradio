

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_FILTER_IIR_FILTER_CCD_PYTHON_HPP
#define INCLUDED_GR_FILTER_IIR_FILTER_CCD_PYTHON_HPP

#include <gnuradio/filter/iir_filter_ccd.h>

void bind_iir_filter_ccd(py::module& m)
{
    using iir_filter_ccd    = gr::filter::iir_filter_ccd;


    py::class_<iir_filter_ccd,gr::sync_block,
        std::shared_ptr<iir_filter_ccd>>(m, "iir_filter_ccd")

        .def(py::init(&iir_filter_ccd::make),
           py::arg("fftaps"), 
           py::arg("fbtaps"), 
           py::arg("oldstyle") = true 
        )
        

        .def("set_taps",&iir_filter_ccd::set_taps,
            py::arg("fftaps"), 
            py::arg("fbtaps") 
        )
        .def("to_basic_block",[](std::shared_ptr<iir_filter_ccd> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_FILTER_IIR_FILTER_CCD_PYTHON_HPP */
