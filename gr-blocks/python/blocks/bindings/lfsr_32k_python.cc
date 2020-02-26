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

#include <gnuradio/blocks/lfsr_32k.h>

void bind_lfsr_32k(py::module& m)
{
    using lfsr_32k    = gr::blocks::lfsr_32k;


    py::class_<lfsr_32k,
        std::shared_ptr<lfsr_32k>>(m, "lfsr_32k")

        .def(py::init<>())
        .def(py::init<gr::blocks::lfsr_32k const &>(),           py::arg("arg0") 
        )

        .def("reset",&lfsr_32k::reset)
        .def("next_bit",&lfsr_32k::next_bit)
        .def("next_byte",&lfsr_32k::next_byte)
        .def("next_short",&lfsr_32k::next_short)
        ;


} 
