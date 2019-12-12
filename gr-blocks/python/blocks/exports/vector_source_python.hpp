//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_BLOCKS_VECTOR_SOURCE_PYTHON_HPP
#define INCLUDED_BLOCKS_VECTOR_SOURCE_PYTHON_HPP

// #pragma once

#include <gnuradio/tags.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/sync_block.h>

#if 1
template<typename T>
void export_vector_source_template(py::module& m, const char *classname)
{
    // using vector_source      = gr::blocks::vector_source<T>;

    
    py::class_<gr::blocks::vector_source<T>, gr::sync_block, std::shared_ptr<gr::blocks::vector_source<T>>>(m, classname)
        // static sptr make(const std::vector<T>& data,
        //                  bool repeat = false,
        //                  unsigned int vlen = 1,
        //                  const std::vector<tag_t>& tags = std::vector<tag_t>());
        .def(py::init(&gr::blocks::vector_source<T>::make),py::arg("data"), py::arg("repeat")=false, py::arg("vlen")=1, py::arg("tags")=std::vector<gr::tag_t>())
        .def("to_basic_block",&gr::blocks::vector_source<T>::to_basic_block)

        // With these templated classes, the additional class members cause the compiler error
        //  error: expected ‘;’ before ‘)’ token
        //  .def("rewind", &gr::blocks::vector_source<T>::rewind))
        //                                                       ^


        // // virtual void rewind() = 0;
        // .def("rewind", &gr::blocks::vector_source<T>::rewind))
        // virtual void set_data(const std::vector<T>& data,
        // //                       const std::vector<tag_t>& tags = std::vector<tag_t>()) = 0;
        // .def("set_data", &gr::blocks::vector_source<T>::set_data))
        // // virtual void set_repeat(bool repeat) = 0;
        // .def("set_repeat", &gr::blocks::vector_source<T>::set_repeat))

        ;
} 

void export_vector_source(py::module& m)
{
    export_vector_source_template<std::uint8_t>(m,"vector_source_b");
    export_vector_source_template<std::int16_t>(m,"vector_source_s");
    export_vector_source_template<std::int32_t>(m,"vector_source_i");
    export_vector_source_template<float>(m,"vector_source_f");
    export_vector_source_template<gr_complex>(m,"vector_source_c");
}

#else
// Brute force it
void export_vector_source(py::module& m)
{
    using vector_source_b      = gr::blocks::vector_source_b;

    py::class_<vector_source_b, std::shared_ptr<vector_source_b>>(m, "vector_source_b")
        // static sptr make(const std::vector<T>& data,
        //                  bool repeat = false,
        //                  unsigned int vlen = 1,
        //                  const std::vector<tag_t>& tags = std::vector<tag_t>());
        .def(py::init(&(gr::blocks::vector_source_b::make)))
        // virtual void rewind() = 0;
        // .def("rewind", &gr::blocks::vector_source_b::rewind))
        // // virtual void set_data(const std::vector<T>& data,
        // //                       const std::vector<tag_t>& tags = std::vector<tag_t>()) = 0;
        // .def("set_data", &gr::blocks::vector_source_b::set_data))
        // // virtual void set_repeat(bool repeat) = 0;
        // .def("set_repeat", &gr::blocks::vector_source_b::set_repeat))

        ;
} 
#endif

#endif /* INCLUDED_BLOCKS_VECTOR_SOURCE_PYTHON_HPP */
