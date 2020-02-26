/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/blocks/sub.h>

template<typename T>
void bind_sub_template(py::module& m, const char *classname)
{
    using sub      = gr::blocks::sub<T>;

    py::class_<sub, gr::sync_block, std::shared_ptr<sub>>(m, classname)
        .def(py::init(&gr::blocks::sub<T>::make),
            py::arg("vlen") = 1
        )

        .def("to_basic_block",[](std::shared_ptr<sub> p){
            return p->to_basic_block();
        })
        ;
} 

void bind_sub(py::module& m)
{
    bind_sub_template<std::int16_t>(m,"sub_ss");
    bind_sub_template<std::int32_t>(m,"sub_ii");
    bind_sub_template<float>(m,"sub_ff");
    bind_sub_template<gr_complex>(m,"sub_cc");

} 

