//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_GR_IO_SIGNATURE_PYTHON_HPP
#define INCLUDED_GR_IO_SIGNATURE_PYTHON_HPP

// #pragma once

#include <gnuradio/io_signature.h>

void export_io_signature(py::module& m)
{
    using io_signature      = gr::io_signature;
    py::class_<io_signature, std::shared_ptr<io_signature>>(m, "io_signature")    
        // io_signature(int min_streams,
        //              int max_streams,
        //              const std::vector<int>& sizeof_stream_items);
        .def(py::init(&io_signature::make))
        .def("make", &io_signature::make)
        .def("make2", &io_signature::make2)
        .def("make3", &io_signature::make3)
        .def("makev", &io_signature::makev)
        // int min_streams() const { return d_min_streams; }
        .def("min_streams", &io_signature::min_streams)
        // int max_streams() const { return d_max_streams; }
        .def("max_streams", &io_signature::max_streams)
        // int sizeof_stream_item(int index) const;
        .def("sizeof_stream_item", &io_signature::sizeof_stream_item)
        // std::vector<int> sizeof_stream_items() const;
        .def("sizeof_stream_items", &io_signature::sizeof_stream_items)
        ;
} 

#endif /* INCLUDED_UHD_USRP_MULTI_USRP_PYTHON_HPP */
