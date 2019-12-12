//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_GR_BLOCK_PYTHON_HPP
#define INCLUDED_GR_BLOCK_PYTHON_HPP

// #pragma once

#include <gnuradio/block.h>

void export_block(py::module& m)
{
    using block      = gr::block;
    // py::class_<boost::enable_shared_from_this<block>>(m,"enable_shared_from_this");

    py::class_<block, gr::basic_block, std::shared_ptr<block>>(m, "block")
            .def("history", &block::history)
            .def("set_history", &block::set_history)

        ;
} 

#endif /* INCLUDED_UHD_USRP_MULTI_USRP_PYTHON_HPP */
