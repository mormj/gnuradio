/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/msg_accepter.h>
// pydoc.h is automatically generated in the build directory
#include <msg_accepter_pydoc.h>

void bind_msg_accepter(py::module& m)
{
    using msg_accepter    = gr::msg_accepter;


    py::class_<msg_accepter, gr::messages::msg_accepter,
        std::shared_ptr<msg_accepter>>(m, "msg_accepter", D(msg_accepter))

        .def(py::init<>(),D(msg_accepter,msg_accepter,0))
        .def(py::init<gr::msg_accepter const &>(),           py::arg("arg0"),
           D(msg_accepter,msg_accepter,1)
        )


        .def("post",&msg_accepter::post,
            py::arg("which_port"),
            py::arg("msg"),
            D(msg_accepter,post)
        )

        ;


} 
