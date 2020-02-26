/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/blocks/pdu_filter.h>

void bind_pdu_filter(py::module& m)
{
    using pdu_filter    = gr::blocks::pdu_filter;


    py::class_<pdu_filter,gr::block,
        std::shared_ptr<pdu_filter>>(m, "pdu_filter")

        .def(py::init(&pdu_filter::make),
           py::arg("k"), 
           py::arg("v"), 
           py::arg("invert") = false 
        )
        

        .def("set_key",&pdu_filter::set_key,
            py::arg("key") 
        )
        .def("set_val",&pdu_filter::set_val,
            py::arg("val") 
        )
        .def("set_inversion",&pdu_filter::set_inversion,
            py::arg("invert") 
        )
        .def("to_basic_block",[](std::shared_ptr<pdu_filter> p){
            return p->to_basic_block();
        })
        ;


} 
