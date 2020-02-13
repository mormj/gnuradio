

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_HEADER_FORMAT_COUNTER_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_HEADER_FORMAT_COUNTER_PYTHON_HPP

#include <gnuradio/digital/header_format_counter.h>

void bind_header_format_counter(py::module& m)
{
    using header_format_counter    = gr::digital::header_format_counter;


    py::class_<header_format_counter,gr::digital::header_format_default,
        std::shared_ptr<header_format_counter>>(m, "header_format_counter")

        // The make function returns a _base pointer, which is incompatible according to pybind11
        // .def(py::init(&header_format_counter::make),
        //    py::arg("access_code"), 
        //    py::arg("threshold"), 
        //    py::arg("bps") 
        // )

        .def_static("make",&header_format_counter::make,
           py::arg("access_code"), 
           py::arg("threshold"), 
           py::arg("bps") 
        )

        .def("format",&header_format_counter::format,
            py::arg("nbytes_in"), 
            py::arg("input"), 
            py::arg("output"), 
            py::arg("info") 
        )
        .def("header_nbits",&header_format_counter::header_nbits)
        ;


} 

#endif /* INCLUDED_GR_DIGITAL_HEADER_FORMAT_COUNTER_PYTHON_HPP */
