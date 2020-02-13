

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_DIFF_ENCODER_BB_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_DIFF_ENCODER_BB_PYTHON_HPP

#include <gnuradio/digital/diff_encoder_bb.h>

void bind_diff_encoder_bb(py::module& m)
{
    using diff_encoder_bb    = gr::digital::diff_encoder_bb;


    py::class_<diff_encoder_bb,gr::sync_block,
        std::shared_ptr<diff_encoder_bb>>(m, "diff_encoder_bb")

        .def(py::init(&diff_encoder_bb::make),
           py::arg("modulus") 
        )
        

        .def("to_basic_block",[](std::shared_ptr<diff_encoder_bb> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_DIGITAL_DIFF_ENCODER_BB_PYTHON_HPP */
