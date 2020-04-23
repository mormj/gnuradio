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

#include <gnuradio/blocks/ctrlport_probe_c.h>
// pydoc.h is automatically generated in the build directory
#include <ctrlport_probe_c_pydoc.h>

void bind_ctrlport_probe_c(py::module& m)
{

    using ctrlport_probe_c = ::gr::blocks::ctrlport_probe_c;


    py::class_<ctrlport_probe_c,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<ctrlport_probe_c>>(
        m, "ctrlport_probe_c", D(ctrlport_probe_c))

        .def(py::init(&ctrlport_probe_c::make),
             py::arg("id"),
             py::arg("desc"),
             D(ctrlport_probe_c, make))


        .def("get", &ctrlport_probe_c::get, D(ctrlport_probe_c, get))

        ;
}
