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

#include <gnuradio/fxpt_vco.h>
// pydoc.h is automatically generated in the build directory
#include <fxpt_vco_pydoc.h>

void bind_fxpt_vco(py::module& m)
{

    using fxpt_vco = ::gr::fxpt_vco;


    py::class_<fxpt_vco, std::shared_ptr<fxpt_vco>>(m, "fxpt_vco", D(fxpt_vco))

        .def(py::init<>(), D(fxpt_vco, fxpt_vco, 0))
        .def(py::init<gr::fxpt_vco const&>(), py::arg("arg0"), D(fxpt_vco, fxpt_vco, 1))


        .def("set_phase", &fxpt_vco::set_phase, py::arg("angle"), D(fxpt_vco, set_phase))


        .def("adjust_phase",
             &fxpt_vco::adjust_phase,
             py::arg("delta_phase"),
             D(fxpt_vco, adjust_phase))


        .def("get_phase", &fxpt_vco::get_phase, D(fxpt_vco, get_phase))


        .def("sincos",
             (void (fxpt_vco::*)(float*, float*) const) & fxpt_vco::sincos,
             py::arg("sinx"),
             py::arg("cosx"),
             D(fxpt_vco, sincos, 0))


        .def("sincos",
             (void (fxpt_vco::*)(gr_complex*, float const*, int, float, float)) &
                 fxpt_vco::sincos,
             py::arg("output"),
             py::arg("input"),
             py::arg("noutput_items"),
             py::arg("k"),
             py::arg("ampl") = 1.,
             D(fxpt_vco, sincos, 1))


        .def("cos",
             (void (fxpt_vco::*)(float*, float const*, int, float, float)) &
                 fxpt_vco::cos,
             py::arg("output"),
             py::arg("input"),
             py::arg("noutput_items"),
             py::arg("k"),
             py::arg("ampl") = 1.,
             D(fxpt_vco, cos, 0))


        .def("cos", (float (fxpt_vco::*)() const) & fxpt_vco::cos, D(fxpt_vco, cos, 1))


        .def("sin", &fxpt_vco::sin, D(fxpt_vco, sin))

        ;
}
