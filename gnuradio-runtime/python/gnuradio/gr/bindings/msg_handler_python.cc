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

#include <gnuradio/msg_handler.h>
// pydoc.h is automatically generated in the build directory
#include <msg_handler_pydoc.h>

void bind_msg_handler(py::module& m)
{
    using msg_handler    = gr::msg_handler;


    py::class_<msg_handler,
        std::shared_ptr<msg_handler>>(m, "msg_handler", D(msg_handler))

        .def(py::init<>(),D(msg_handler,msg_handler,0))
        .def(py::init<gr::msg_handler const &>(),           py::arg("arg0"),
           D(msg_handler,msg_handler,1)
        )


        .def("handle",&msg_handler::handle,
            py::arg("msg"),
            D(msg_handler,handle)
        )
        ;


} 
