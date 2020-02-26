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

#include <gnuradio/blocks/divide.h>

template<typename T>
void bind_divide_template(py::module& m, const char *classname)
{
    using divide      = gr::blocks::divide<T>;

    py::class_<divide, gr::sync_block, std::shared_ptr<divide>>(m, classname)
        .def(py::init(&gr::blocks::divide<T>::make),
            py::arg("vlen") = 1
        )

        .def("to_basic_block",[](std::shared_ptr<divide> p){
            return p->to_basic_block();
        })
        ;
} 

void bind_divide(py::module& m)
{
    bind_divide_template<std::int16_t>(m,"divide_ss");
    bind_divide_template<std::int32_t>(m,"divide_ii");
    bind_divide_template<float>(m,"divide_ff");
    bind_divide_template<gr_complex>(m,"divide_cc");
} 

