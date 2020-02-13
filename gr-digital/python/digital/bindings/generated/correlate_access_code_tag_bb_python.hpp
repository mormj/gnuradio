

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_CORRELATE_ACCESS_CODE_TAG_BB_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_CORRELATE_ACCESS_CODE_TAG_BB_PYTHON_HPP

#include <gnuradio/digital/correlate_access_code_tag_bb.h>

void bind_correlate_access_code_tag_bb(py::module& m)
{
    using correlate_access_code_tag_bb    = gr::digital::correlate_access_code_tag_bb;


    py::class_<correlate_access_code_tag_bb,gr::sync_block,
        std::shared_ptr<correlate_access_code_tag_bb>>(m, "correlate_access_code_tag_bb")

        .def(py::init(&correlate_access_code_tag_bb::make),
           py::arg("access_code"), 
           py::arg("threshold"), 
           py::arg("tag_name") 
        )
        

        .def("set_access_code",&correlate_access_code_tag_bb::set_access_code,
            py::arg("access_code") 
        )
        .def("set_threshold",&correlate_access_code_tag_bb::set_threshold,
            py::arg("threshold") 
        )
        .def("set_tagname",&correlate_access_code_tag_bb::set_tagname,
            py::arg("tagname") 
        )
        .def("to_basic_block",[](std::shared_ptr<correlate_access_code_tag_bb> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_DIGITAL_CORRELATE_ACCESS_CODE_TAG_BB_PYTHON_HPP */
