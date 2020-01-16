//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_BLOCKS_NULL_SINK_PYTHON_HPP
#define INCLUDED_BLOCKS_NULL_SINK_PYTHON_HPP

// #pragma once

#include <gnuradio/blocks/null_sink.h>
#include <gnuradio/sync_block.h>


void bind_null_sink(py::module& m)
{
    using null_sink      = gr::blocks::null_sink;

    py::class_<null_sink, gr::sync_block, std::shared_ptr<null_sink>>(m, "null_sink")
        .def(py::init(&null_sink::make))
        .def("to_basic_block",[](std::shared_ptr<null_sink> p){
            return p->to_basic_block();
        })
        ;
} 

// namespace pybind11 {
//     template<> struct polymorphic_type_hook<gr::blocks::null_sink> {
//         static const void *get(const gr::blocks::null_sink *src, const std::type_info*& type) {
//             // note that src may be nullptr
//             if (src) {
//                 type = &typeid(gr::basic_block);
//                 return dynamic_cast<const gr::basic_block*>(src);
//             }
//             return src;
//         }
//     };

//     template<> struct polymorphic_type_hook<gr::basic_block> {
//         static const void *get(const gr::basic_block *src, const std::type_info*& type) {
//             // note that src may be nullptr
//             if (src) {
//                 type = &typeid(gr::blocks::null_sink);
//                 return dynamic_cast<const gr::blocks::null_sink*>(src);
//             }
//             return src;
//         }
//     };
// } // namespace pybind11

// namespace pybind11 {
//     template<> struct polymorphic_type_hook<gr::basic_block> {
//         static const void *get(const gr::basic_block *src, const std::type_info*& type) {
//             // note that src may be nullptr
//             if (src) {
//                 type = &typeid(gr::blocks::null_sink);
//                 return dynamic_cast<const gr::blocks::null_sink*>(src);
//             }
//             return src;
//         }
//     };
// } // namespace pybind11

#endif /* INCLUDED_BLOCKS_null_sink_PYTHON_HPP */
