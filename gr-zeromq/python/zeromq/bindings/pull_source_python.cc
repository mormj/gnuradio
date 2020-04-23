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

#include <gnuradio/zeromq/pull_source.h>
// pydoc.h is automatically generated in the build directory
#include <pull_source_pydoc.h>

void bind_pull_source(py::module& m)
{

    using pull_source = ::gr::zeromq::pull_source;


    py::class_<pull_source,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<pull_source>>(m, "pull_source", D(pull_source))

        .def(py::init(&pull_source::make),
             py::arg("itemsize"),
             py::arg("vlen"),
             py::arg("address"),
             py::arg("timeout") = 100,
             py::arg("pass_tags") = false,
             py::arg("hwm") = -1,
             D(pull_source, make))


        .def("last_endpoint", &pull_source::last_endpoint, D(pull_source, last_endpoint))

        ;
}
