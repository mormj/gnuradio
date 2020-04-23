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

#include <gnuradio/blocks/tag_gate.h>
// pydoc.h is automatically generated in the build directory
#include <tag_gate_pydoc.h>

void bind_tag_gate(py::module& m)
{

    using tag_gate = ::gr::blocks::tag_gate;


    py::class_<tag_gate,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<tag_gate>>(m, "tag_gate", D(tag_gate))

        .def(py::init(&tag_gate::make),
             py::arg("item_size"),
             py::arg("propagate_tags") = false,
             D(tag_gate, make))


        .def("set_propagation",
             &tag_gate::set_propagation,
             py::arg("propagate_tags"),
             D(tag_gate, set_propagation))


        .def("set_single_key",
             &tag_gate::set_single_key,
             py::arg("single_key"),
             D(tag_gate, set_single_key))


        .def("single_key", &tag_gate::single_key, D(tag_gate, single_key))

        ;
}
