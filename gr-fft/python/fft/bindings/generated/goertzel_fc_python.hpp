

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_FFT_GOERTZEL_FC_PYTHON_HPP
#define INCLUDED_GR_FFT_GOERTZEL_FC_PYTHON_HPP

#include <gnuradio/fft/goertzel_fc.h>

void bind_goertzel_fc(py::module& m)
{
    using goertzel_fc    = gr::fft::goertzel_fc;


    py::class_<goertzel_fc,gr::sync_decimator,
        std::shared_ptr<goertzel_fc>>(m, "goertzel_fc")

        .def(py::init(&goertzel_fc::make),
           py::arg("rate"), 
           py::arg("len"), 
           py::arg("freq") 
        )
        

        .def("set_freq",&goertzel_fc::set_freq,
            py::arg("freq") 
        )
        .def("set_rate",&goertzel_fc::set_rate,
            py::arg("rate") 
        )
        .def("freq",&goertzel_fc::freq)
        .def("rate",&goertzel_fc::rate)
        .def("to_basic_block",[](std::shared_ptr<goertzel_fc> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_FFT_GOERTZEL_FC_PYTHON_HPP */
