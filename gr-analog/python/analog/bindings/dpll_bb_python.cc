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

#include <gnuradio/analog/dpll_bb.h>
// pydoc.h is automatically generated in the build directory
#include <dpll_bb_pydoc.h>

void bind_dpll_bb(py::module& m)
{

    using dpll_bb = ::gr::analog::dpll_bb;


    py::class_<dpll_bb,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<dpll_bb>>(m, "dpll_bb", D(dpll_bb))

        .def(py::init(&dpll_bb::make),
             py::arg("period"),
             py::arg("gain"),
             D(dpll_bb, make))


        .def("set_gain", &dpll_bb::set_gain, py::arg("gain"), D(dpll_bb, set_gain))


        .def("set_decision_threshold",
             &dpll_bb::set_decision_threshold,
             py::arg("thresh"),
             D(dpll_bb, set_decision_threshold))


        .def("gain", &dpll_bb::gain, D(dpll_bb, gain))


        .def("freq", &dpll_bb::freq, D(dpll_bb, freq))


        .def("phase", &dpll_bb::phase, D(dpll_bb, phase))


        .def("decision_threshold",
             &dpll_bb::decision_threshold,
             D(dpll_bb, decision_threshold))

        ;
}
