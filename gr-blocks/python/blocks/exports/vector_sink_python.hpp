//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_BLOCKS_VECTOR_SINK_PYTHON_HPP
#define INCLUDED_BLOCKS_VECTOR_SINK_PYTHON_HPP

// #pragma once

#include <gnuradio/tags.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/sync_block.h>

template<typename T>
void export_vector_sink_template(py::module& m, const char *classname)
{
    using vector_sink      = gr::blocks::vector_sink<T>;

    py::class_<vector_sink, gr::sync_block, std::shared_ptr<vector_sink>>(m, classname)
        .def(py::init(&gr::blocks::vector_sink<T>::make), py::arg("vlen")=1, py::arg("reserve_items")=1024)
        .def("to_basic_block",&vector_sink::to_basic_block)
        .def("reset",&vector_sink::reset)
        .def("data",&vector_sink::data)
        .def("tags",&vector_sink::tags)
        ;
} 

void export_vector_sink(py::module& m)
{
    export_vector_sink_template<std::uint8_t>(m,"vector_sink_b");
    export_vector_sink_template<std::int16_t>(m,"vector_sink_s");
    export_vector_sink_template<std::int32_t>(m,"vector_sink_i");
    export_vector_sink_template<float>(m,"vector_sink_f");
    export_vector_sink_template<gr_complex>(m,"vector_sink_c");
}


#endif /* INCLUDED_BLOCKS_vector_sink_PYTHON_HPP */
