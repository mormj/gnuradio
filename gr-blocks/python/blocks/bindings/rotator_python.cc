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

#include <gnuradio/blocks/rotator.h>

void bind_rotator(py::module& m)
{
    using rotator    = gr::blocks::rotator;


    py::class_<rotator,
        std::shared_ptr<rotator>>(m, "rotator")

        .def(py::init<>())
        .def(py::init<gr::blocks::rotator const &>(),           py::arg("arg0") 
        )

        .def("set_phase",&rotator::set_phase,
            py::arg("phase") 
        )
        .def("set_phase_incr",&rotator::set_phase_incr,
            py::arg("incr") 
        )
        .def("rotate",&rotator::rotate,
            py::arg("in") 
        )
        .def("rotateN",&rotator::rotateN,
            py::arg("out"), 
            py::arg("in"), 
            py::arg("n") 
        )
        ;


} 
