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

#include <gnuradio/blocks/plateau_detector_fb.h>

void bind_plateau_detector_fb(py::module& m)
{
    using plateau_detector_fb    = gr::blocks::plateau_detector_fb;


    py::class_<plateau_detector_fb,gr::block,
        std::shared_ptr<plateau_detector_fb>>(m, "plateau_detector_fb")

        .def(py::init(&plateau_detector_fb::make),
           py::arg("max_len"), 
           py::arg("threshold") = 0.90000000000000002 
        )
        

        .def("set_threshold",&plateau_detector_fb::set_threshold,
            py::arg("threshold") 
        )
        .def("threshold",&plateau_detector_fb::threshold)
        .def("to_basic_block",[](std::shared_ptr<plateau_detector_fb> p){
            return p->to_basic_block();
        })
        ;


} 
