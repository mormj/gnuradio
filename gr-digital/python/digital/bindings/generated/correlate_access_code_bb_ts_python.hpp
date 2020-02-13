

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_CORRELATE_ACCESS_CODE_BB_TS_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_CORRELATE_ACCESS_CODE_BB_TS_PYTHON_HPP

#include <gnuradio/digital/correlate_access_code_bb_ts.h>

void bind_correlate_access_code_bb_ts(py::module& m)
{
    using correlate_access_code_bb_ts    = gr::digital::correlate_access_code_bb_ts;


    py::class_<correlate_access_code_bb_ts,gr::block,
        std::shared_ptr<correlate_access_code_bb_ts>>(m, "correlate_access_code_bb_ts")

        .def(py::init(&correlate_access_code_bb_ts::make),
           py::arg("access_code"), 
           py::arg("threshold"), 
           py::arg("tag_name") 
        )
        

        .def("set_access_code",&correlate_access_code_bb_ts::set_access_code,
            py::arg("access_code") 
        )
        .def("access_code",&correlate_access_code_bb_ts::access_code)
        .def("to_basic_block",[](std::shared_ptr<correlate_access_code_bb_ts> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_DIGITAL_CORRELATE_ACCESS_CODE_BB_TS_PYTHON_HPP */
