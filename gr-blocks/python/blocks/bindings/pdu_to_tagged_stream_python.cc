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

#include <gnuradio/blocks/pdu_to_tagged_stream.h>
// pydoc.h is automatically generated in the build directory
#include <pdu_to_tagged_stream_pydoc.h>

void bind_pdu_to_tagged_stream(py::module& m)
{

    using pdu_to_tagged_stream = ::gr::blocks::pdu_to_tagged_stream;


    py::class_<pdu_to_tagged_stream,
               gr::tagged_stream_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<pdu_to_tagged_stream>>(
        m, "pdu_to_tagged_stream", D(pdu_to_tagged_stream))

        .def(py::init(&pdu_to_tagged_stream::make),
             py::arg("type"),
             py::arg("lengthtagname") = "packet_len",
             D(pdu_to_tagged_stream, make))


        ;


    py::module m_pdu = m.def_submodule("pdu");
}
