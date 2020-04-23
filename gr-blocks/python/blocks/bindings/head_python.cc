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

#include <gnuradio/blocks/head.h>
// pydoc.h is automatically generated in the build directory
#include <head_pydoc.h>

void bind_head(py::module& m)
{

    using head = ::gr::blocks::head;


    py::class_<head, gr::sync_block, gr::block, gr::basic_block, std::shared_ptr<head>>(
        m, "head", D(head))

        .def(py::init(&head::make),
             py::arg("sizeof_stream_item"),
             py::arg("nitems"),
             D(head, make))


        .def("reset", &head::reset, D(head, reset))


        .def("set_length", &head::set_length, py::arg("nitems"), D(head, set_length))

        ;
}
