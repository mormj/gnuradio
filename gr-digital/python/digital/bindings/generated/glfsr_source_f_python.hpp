

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_GLFSR_SOURCE_F_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_GLFSR_SOURCE_F_PYTHON_HPP

#include <gnuradio/digital/glfsr_source_f.h>

void bind_glfsr_source_f(py::module& m)
{
    using glfsr_source_f    = gr::digital::glfsr_source_f;


    py::class_<glfsr_source_f,gr::sync_block,
        std::shared_ptr<glfsr_source_f>>(m, "glfsr_source_f")

        .def(py::init(&glfsr_source_f::make),
           py::arg("degree"), 
           py::arg("repeat") = true, 
           py::arg("mask") = 0, 
           py::arg("seed") = 1 
        )
        

        .def("period",&glfsr_source_f::period)
        .def("mask",&glfsr_source_f::mask)
        .def("to_basic_block",[](std::shared_ptr<glfsr_source_f> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_DIGITAL_GLFSR_SOURCE_F_PYTHON_HPP */
