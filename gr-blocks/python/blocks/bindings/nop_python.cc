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

#include <gnuradio/blocks/nop.h>

void bind_nop(py::module& m)
{
    using nop    = gr::blocks::nop;


    py::class_<nop,gr::block,
        std::shared_ptr<nop>>(m, "nop")

        .def(py::init(&nop::make),
           py::arg("sizeof_stream_item") 
        )
        

        .def("nmsgs_received",&nop::nmsgs_received)
        .def("ctrlport_test",&nop::ctrlport_test)
        .def("set_ctrlport_test",&nop::set_ctrlport_test,
            py::arg("x") 
        )
        .def("to_basic_block",[](std::shared_ptr<nop> p){
            return p->to_basic_block();
        })
        ;


} 
