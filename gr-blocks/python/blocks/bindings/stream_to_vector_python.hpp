//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_BLOCKS_STREAM_TO_VECTOR_PYTHON_HPP
#define INCLUDED_BLOCKS_STREAM_TO_VECTOR_PYTHON_HPP

// #pragma once

#include <gnuradio/blocks/stream_to_vector.h>
#include <gnuradio/sync_block.h>


void bind_stream_to_vector(py::module& m)
{
    using stream_to_vector      = gr::blocks::stream_to_vector;

    py::class_<stream_to_vector, gr::sync_block, std::shared_ptr<stream_to_vector>>(m, "stream_to_vector")
        .def(py::init(&stream_to_vector::make),py::arg("itemsize"), py::arg("nitems_per_block"))
        .def("to_basic_block",&stream_to_vector::to_basic_block)
        ;
} 

#endif /* INCLUDED_BLOCKS_STREAM_TO_VECTOR_PYTHON_HPP */
