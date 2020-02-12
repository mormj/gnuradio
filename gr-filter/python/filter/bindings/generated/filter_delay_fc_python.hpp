

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_FILTER_FILTER_DELAY_FC_PYTHON_HPP
#define INCLUDED_GR_FILTER_FILTER_DELAY_FC_PYTHON_HPP

#include <gnuradio/filter/filter_delay_fc.h>

void bind_filter_delay_fc(py::module& m)
{
    using filter_delay_fc    = gr::filter::filter_delay_fc;


    py::class_<filter_delay_fc,gr::sync_block,
        std::shared_ptr<filter_delay_fc>>(m, "filter_delay_fc")

        .def(py::init(&filter_delay_fc::make),
           py::arg("taps") 
        )
        

        .def("to_basic_block",[](std::shared_ptr<filter_delay_fc> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_FILTER_FILTER_DELAY_FC_PYTHON_HPP */
