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

#include <gnuradio/trellis/constellation_metrics_cf.h>
// pydoc.h is automatically generated in the build directory
#include <constellation_metrics_cf_pydoc.h>

void bind_constellation_metrics_cf(py::module& m)
{

    using constellation_metrics_cf = ::gr::trellis::constellation_metrics_cf;


    py::class_<constellation_metrics_cf,
               gr::block,
               gr::basic_block,
               std::shared_ptr<constellation_metrics_cf>>(
        m, "constellation_metrics_cf", D(constellation_metrics_cf))

        .def(py::init(&constellation_metrics_cf::make),
             py::arg("constellation"),
             py::arg("TYPE"),
             D(constellation_metrics_cf, make))


        ;
}
