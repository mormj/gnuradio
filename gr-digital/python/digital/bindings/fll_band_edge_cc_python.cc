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

#include <gnuradio/digital/fll_band_edge_cc.h>
// pydoc.h is automatically generated in the build directory
#include <fll_band_edge_cc_pydoc.h>

void bind_fll_band_edge_cc(py::module& m)
{

    using fll_band_edge_cc = ::gr::digital::fll_band_edge_cc;


    py::class_<fll_band_edge_cc,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<fll_band_edge_cc>>(
        m, "fll_band_edge_cc", D(fll_band_edge_cc))

        .def(py::init(&fll_band_edge_cc::make),
             py::arg("samps_per_sym"),
             py::arg("rolloff"),
             py::arg("filter_size"),
             py::arg("bandwidth"),
             D(fll_band_edge_cc, make))


        .def("set_samples_per_symbol",
             &fll_band_edge_cc::set_samples_per_symbol,
             py::arg("sps"),
             D(fll_band_edge_cc, set_samples_per_symbol))


        .def("set_rolloff",
             &fll_band_edge_cc::set_rolloff,
             py::arg("rolloff"),
             D(fll_band_edge_cc, set_rolloff))


        .def("set_filter_size",
             &fll_band_edge_cc::set_filter_size,
             py::arg("filter_size"),
             D(fll_band_edge_cc, set_filter_size))


        .def("samples_per_symbol",
             &fll_band_edge_cc::samples_per_symbol,
             D(fll_band_edge_cc, samples_per_symbol))


        .def("rolloff", &fll_band_edge_cc::rolloff, D(fll_band_edge_cc, rolloff))


        .def("filter_size",
             &fll_band_edge_cc::filter_size,
             D(fll_band_edge_cc, filter_size))


        .def("print_taps", &fll_band_edge_cc::print_taps, D(fll_band_edge_cc, print_taps))

        ;
}
