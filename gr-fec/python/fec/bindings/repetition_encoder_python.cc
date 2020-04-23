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

#include <gnuradio/fec/repetition_encoder.h>
// pydoc.h is automatically generated in the build directory
#include <repetition_encoder_pydoc.h>

void bind_repetition_encoder(py::module& m)
{


    py::module m_code = m.def_submodule("code");

    using repetition_encoder = ::gr::fec::code::repetition_encoder;


    py::class_<repetition_encoder,
               gr::fec::generic_encoder,
               std::shared_ptr<repetition_encoder>>(
        m_code, "repetition_encoder", D(code, repetition_encoder))

        .def_static("make",
                    &repetition_encoder::make,
                    py::arg("frame_size"),
                    py::arg("rep"),
                    D(code, repetition_encoder, make))


        .def("set_frame_size",
             &repetition_encoder::set_frame_size,
             py::arg("frame_size"),
             D(code, repetition_encoder, set_frame_size))


        .def("rate", &repetition_encoder::rate, D(code, repetition_encoder, rate))

        ;
}
