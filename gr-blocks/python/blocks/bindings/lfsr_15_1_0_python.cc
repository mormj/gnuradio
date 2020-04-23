/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/blocks/lfsr_15_1_0.h>
// pydoc.h is automatically generated in the build directory
#include <lfsr_15_1_0_pydoc.h>

void bind_lfsr_15_1_0(py::module& m)
{

    using lfsr_15_1_0 = ::gr::blocks::lfsr_15_1_0;


    py::class_<lfsr_15_1_0, std::shared_ptr<lfsr_15_1_0>>(
        m, "lfsr_15_1_0", D(lfsr_15_1_0))

        .def(py::init<>(), D(lfsr_15_1_0, lfsr_15_1_0, 0))
        .def(py::init<gr::blocks::lfsr_15_1_0 const&>(),
             py::arg("arg0"),
             D(lfsr_15_1_0, lfsr_15_1_0, 1))


        .def("reset", &lfsr_15_1_0::reset, D(lfsr_15_1_0, reset))


        .def("next_bit", &lfsr_15_1_0::next_bit, D(lfsr_15_1_0, next_bit))


        .def("next_byte", &lfsr_15_1_0::next_byte, D(lfsr_15_1_0, next_byte))

        ;
}
