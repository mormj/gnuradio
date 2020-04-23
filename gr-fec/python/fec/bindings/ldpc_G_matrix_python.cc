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

#include <gnuradio/fec/ldpc_G_matrix.h>
// pydoc.h is automatically generated in the build directory
#include <ldpc_G_matrix_pydoc.h>

void bind_ldpc_G_matrix(py::module& m)
{


    py::module m_code = m.def_submodule("code");

    using ldpc_G_matrix = ::gr::fec::code::ldpc_G_matrix;


    py::class_<ldpc_G_matrix, gr::fec::code::fec_mtrx, std::shared_ptr<ldpc_G_matrix>>(
        m_code, "ldpc_G_matrix", D(code, ldpc_G_matrix))

        .def(py::init(&ldpc_G_matrix::make),
             py::arg("filename"),
             D(code, ldpc_G_matrix, make))


        .def("encode",
             &ldpc_G_matrix::encode,
             py::arg("outbuffer"),
             py::arg("inbuffer"),
             D(code, ldpc_G_matrix, encode))


        .def("decode",
             &ldpc_G_matrix::decode,
             py::arg("outbuffer"),
             py::arg("inbuffer"),
             py::arg("frame_size"),
             py::arg("max_iterations"),
             D(code, ldpc_G_matrix, decode))


        .def("get_base_sptr",
             &ldpc_G_matrix::get_base_sptr,
             D(code, ldpc_G_matrix, get_base_sptr))

        ;
}
