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

#include <gnuradio/blocks/rms_ff.h>
// pydoc.h is automatically generated in the build directory
#include <rms_ff_pydoc.h>

void bind_rms_ff(py::module& m)
{

    using rms_ff = ::gr::blocks::rms_ff;


    py::class_<rms_ff,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<rms_ff>>(m, "rms_ff", D(rms_ff))

        .def(py::init(&rms_ff::make), py::arg("alpha") = 1.0E-4, D(rms_ff, make))


        .def("set_alpha", &rms_ff::set_alpha, py::arg("alpha"), D(rms_ff, set_alpha))

        ;
}
