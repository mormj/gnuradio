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

#include <gnuradio/fec/polar_decoder_sc_list.h>
// pydoc.h is automatically generated in the build directory
#include <polar_decoder_sc_list_pydoc.h>

void bind_polar_decoder_sc_list(py::module& m)
{


    py::module m_code = m.def_submodule("code");

    using polar_decoder_sc_list = ::gr::fec::code::polar_decoder_sc_list;


    py::class_<polar_decoder_sc_list,
               gr::fec::code::polar_decoder_common,
               std::shared_ptr<polar_decoder_sc_list>>(
        m_code, "polar_decoder_sc_list", D(code, polar_decoder_sc_list))

        .def_static("make",
                    &polar_decoder_sc_list::make,
                    py::arg("max_list_size"),
                    py::arg("block_size"),
                    py::arg("num_info_bits"),
                    py::arg("frozen_bit_positions"),
                    py::arg("frozen_bit_values"),
                    D(code, polar_decoder_sc_list, make))


        .def("generic_work",
             &polar_decoder_sc_list::generic_work,
             py::arg("in_buffer"),
             py::arg("out_buffer"),
             D(code, polar_decoder_sc_list, generic_work))

        ;


    py::module m_code_polar = m_code.def_submodule("polar");


    py::module m_polar = m.def_submodule("polar");
}
