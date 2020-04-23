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

#include <gnuradio/analog/pll_freqdet_cf.h>
// pydoc.h is automatically generated in the build directory
#include <pll_freqdet_cf_pydoc.h>

void bind_pll_freqdet_cf(py::module& m)
{

    using pll_freqdet_cf = ::gr::analog::pll_freqdet_cf;


    py::class_<pll_freqdet_cf,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<pll_freqdet_cf>>(m, "pll_freqdet_cf", D(pll_freqdet_cf))

        .def(py::init(&pll_freqdet_cf::make),
             py::arg("loop_bw"),
             py::arg("max_freq"),
             py::arg("min_freq"),
             D(pll_freqdet_cf, make))


        .def("set_loop_bandwidth",
             &pll_freqdet_cf::set_loop_bandwidth,
             py::arg("bw"),
             D(pll_freqdet_cf, set_loop_bandwidth))


        .def("set_damping_factor",
             &pll_freqdet_cf::set_damping_factor,
             py::arg("df"),
             D(pll_freqdet_cf, set_damping_factor))


        .def("set_alpha",
             &pll_freqdet_cf::set_alpha,
             py::arg("alpha"),
             D(pll_freqdet_cf, set_alpha))


        .def("set_beta",
             &pll_freqdet_cf::set_beta,
             py::arg("beta"),
             D(pll_freqdet_cf, set_beta))


        .def("set_frequency",
             &pll_freqdet_cf::set_frequency,
             py::arg("freq"),
             D(pll_freqdet_cf, set_frequency))


        .def("set_phase",
             &pll_freqdet_cf::set_phase,
             py::arg("phase"),
             D(pll_freqdet_cf, set_phase))


        .def("set_min_freq",
             &pll_freqdet_cf::set_min_freq,
             py::arg("freq"),
             D(pll_freqdet_cf, set_min_freq))


        .def("set_max_freq",
             &pll_freqdet_cf::set_max_freq,
             py::arg("freq"),
             D(pll_freqdet_cf, set_max_freq))


        .def("get_loop_bandwidth",
             &pll_freqdet_cf::get_loop_bandwidth,
             D(pll_freqdet_cf, get_loop_bandwidth))


        .def("get_damping_factor",
             &pll_freqdet_cf::get_damping_factor,
             D(pll_freqdet_cf, get_damping_factor))


        .def("get_alpha", &pll_freqdet_cf::get_alpha, D(pll_freqdet_cf, get_alpha))


        .def("get_beta", &pll_freqdet_cf::get_beta, D(pll_freqdet_cf, get_beta))


        .def("get_frequency",
             &pll_freqdet_cf::get_frequency,
             D(pll_freqdet_cf, get_frequency))


        .def("get_phase", &pll_freqdet_cf::get_phase, D(pll_freqdet_cf, get_phase))


        .def("get_min_freq",
             &pll_freqdet_cf::get_min_freq,
             D(pll_freqdet_cf, get_min_freq))


        .def("get_max_freq",
             &pll_freqdet_cf::get_max_freq,
             D(pll_freqdet_cf, get_max_freq))

        ;
}
