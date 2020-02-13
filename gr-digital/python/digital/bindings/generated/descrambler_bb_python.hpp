

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_DESCRAMBLER_BB_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_DESCRAMBLER_BB_PYTHON_HPP

#include <gnuradio/digital/descrambler_bb.h>

void bind_descrambler_bb(py::module& m)
{
    using descrambler_bb    = gr::digital::descrambler_bb;


    py::class_<descrambler_bb,gr::sync_block,
        std::shared_ptr<descrambler_bb>>(m, "descrambler_bb")

        .def(py::init(&descrambler_bb::make),
           py::arg("mask"), 
           py::arg("seed"), 
           py::arg("len") 
        )
        

        .def("to_basic_block",[](std::shared_ptr<descrambler_bb> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_DIGITAL_DESCRAMBLER_BB_PYTHON_HPP */
