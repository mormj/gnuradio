

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_CRC32_BB_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_CRC32_BB_PYTHON_HPP

#include <gnuradio/digital/crc32_bb.h>

void bind_crc32_bb(py::module& m)
{
    using crc32_bb    = gr::digital::crc32_bb;


    py::class_<crc32_bb,gr::tagged_stream_block,
        std::shared_ptr<crc32_bb>>(m, "crc32_bb")

        .def(py::init(&crc32_bb::make),
           py::arg("check") = false, 
           py::arg("lengthtagname") = "packet_len", 
           py::arg("packed") = true 
        )
        

        .def("to_basic_block",[](std::shared_ptr<crc32_bb> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_DIGITAL_CRC32_BB_PYTHON_HPP */
