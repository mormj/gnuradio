//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_GR_HIER_BLOCK2_PYTHON_HPP
#define INCLUDED_GR_HIER_BLOCK2_PYTHON_HPP

// #pragma once

#include <gnuradio/hier_block2.h>
#include <gnuradio/basic_block.h>

void bind_hier_block2(py::module& m)
{
    using hier_block2      = gr::hier_block2;
    using basic_block_sptr = std::shared_ptr<gr::basic_block>;

    py::class_<hier_block2, gr::basic_block, std::shared_ptr<hier_block2>>(m, "hier_block2_pb")     
        
        .def(py::init(&gr::make_hier_block2))
        .def("primitive_connect", (void (hier_block2::*)(basic_block_sptr)) &hier_block2::connect)
        .def("primitive_connect", (void (hier_block2::*)(basic_block_sptr, int, basic_block_sptr, int)) &hier_block2::connect)
        .def("primitive_disconnect", (void (hier_block2::*)(basic_block_sptr)) &hier_block2::disconnect)
        .def("primitive_disconnect", (void (hier_block2::*)(basic_block_sptr, int, basic_block_sptr, int)) &hier_block2::disconnect)
        .def("primitive_msg_connect", (void (hier_block2::*)(basic_block_sptr,
                     pmt::pmt_t,
                     basic_block_sptr,
                     pmt::pmt_t)) &hier_block2::msg_connect)
        .def("primitive_msg_connect", (void (hier_block2::*)(basic_block_sptr,
                     std::string,
                     basic_block_sptr,
                     std::string)) &hier_block2::msg_connect)
        .def("primitive_msg_disconnect", (void (hier_block2::*)(basic_block_sptr,
                     pmt::pmt_t,
                     basic_block_sptr,
                     pmt::pmt_t)) &hier_block2::msg_disconnect)
        .def("primitive_msg_disconnect", (void (hier_block2::*)(basic_block_sptr,
                     std::string,
                     basic_block_sptr,
                     std::string)) &hier_block2::msg_disconnect)
        ;
} 

#endif /* INCLUDED_UHD_USRP_MULTI_USRP_PYTHON_HPP */
