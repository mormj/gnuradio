//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_BLOCKS_MULTIPLY_CONST_PYTHON_HPP
#define INCLUDED_BLOCKS_MULTIPLY_CONST_PYTHON_HPP

// #pragma once

#include <gnuradio/blocks/multiply_const.h>
#include <gnuradio/sync_block.h>

template<typename T>
void export_multiply_const_template(py::module& m, const char *classname)
{
    using multiply_const      = gr::blocks::multiply_const<T>;

    py::class_<multiply_const, gr::sync_block, std::shared_ptr<multiply_const>>(m, classname)
        .def(py::init(&multiply_const::make),py::arg("k"), py::arg("vlen")=1)
        .def("k",&multiply_const::k)
        .def("set_k",&multiply_const::set_k)
        ;
} 

void export_multiply_const(py::module& m)
{
// typedef multiply_const<std::int16_t> multiply_const_ss;
// typedef multiply_const<std::int32_t> multiply_const_ii;
// typedef multiply_const<float> multiply_const_ff;
// typedef multiply_const<gr_complex> multiply_const_cc;
export_multiply_const_template<std::int16_t>(m,"multiply_const_ss");
export_multiply_const_template<std::int32_t>(m,"multiply_const_ii");
export_multiply_const_template<float>(m,"multiply_const_ff");
export_multiply_const_template<gr_complex>(m,"multiply_const_cc");
}

#endif /* INCLUDED_BLOCKS_multiply_const_PYTHON_HPP */
