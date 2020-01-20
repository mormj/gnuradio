//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_GR_BASIC_BLOCK_PYTHON_HPP
#define INCLUDED_GR_BASIC_BLOCK_PYTHON_HPP

// #pragma once

#include <gnuradio/basic_block.h>
#include <gnuradio/msg_accepter.h>
#include <pmt/pmt.h>
// #include <boost/enable_shared_from_this.hpp>

void bind_basic_block(py::module& m)
{
    using basic_block      = gr::basic_block;
    // py::class_<std::enable_shared_from_this<basic_block>>(m,"enable_shared_from_this");

    py::class_<basic_block, std::shared_ptr<basic_block>>(m, "basic_block")
        // pmt::pmt_t message_subscribers(pmt::pmt_t port);
        .def("message_subscribers", &basic_block::message_subscribers)
        // long unique_id() const { return d_unique_id; }
        .def("unique_id", &basic_block::unique_id)
        // long symbolic_id() const { return d_symbolic_id; }
        .def("symbolic_id", &basic_block::symbolic_id)
        // std::string name() const { return d_name; }
        .def("name", &basic_block::name)
        // std::string symbol_name() const { return d_symbol_name; }
        .def("symbol_name", &basic_block::symbol_name)

        // basic_block_sptr to_basic_block();
        .def("to_basic_block", &basic_block::to_basic_block)


        .def("input_signature", &basic_block::input_signature)
        .def("output_signature", &basic_block::output_signature)

        ;
} 

#endif /* INCLUDED_UHD_USRP_MULTI_USRP_PYTHON_HPP */
