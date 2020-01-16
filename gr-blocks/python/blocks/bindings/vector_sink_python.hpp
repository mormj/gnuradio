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

// namespace gr {
// namespace blocks {
//     class 
// virtual std::shared_ptr<gr::basic_block> to_basic_block()
// {
//     return std::enable_shared_from_this<gr::basic_block>::shared_from_this();
// }
// } // namespace blocks
// } // namespace gr

template<typename T>
void bind_vector_sink_template(py::module& m, const char *classname)
{
    using vector_sink      = gr::blocks::vector_sink<T>;

    py::class_<vector_sink, gr::sync_block, std::shared_ptr<vector_sink>>(m, classname)
        .def(py::init(&gr::blocks::vector_sink<T>::make), py::arg("vlen")=1, py::arg("reserve_items")=1024)
        // .def("to_basic_block",&vector_sink::to_basic_block)
        .def("to_basic_block",[](std::shared_ptr<vector_sink> p){
            return p->to_basic_block();
        })
        .def("reset",&vector_sink::reset)
        .def("data",&vector_sink::data)
        .def("tags",&vector_sink::tags)
        ;
} 

void bind_vector_sink(py::module& m)
{
    bind_vector_sink_template<std::uint8_t>(m,"vector_sink_b");
    bind_vector_sink_template<std::int16_t>(m,"vector_sink_s");
    bind_vector_sink_template<std::int32_t>(m,"vector_sink_i");
    bind_vector_sink_template<float>(m,"vector_sink_f");
    bind_vector_sink_template<gr_complex>(m,"vector_sink_c");
}


#endif /* INCLUDED_BLOCKS_vector_sink_PYTHON_HPP */
