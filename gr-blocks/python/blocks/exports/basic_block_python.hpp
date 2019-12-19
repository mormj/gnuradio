//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

/// THIS IS A COPY OF THE MAIN BASIC BLOCK FROM GR -- need to put in shareable locations

#ifndef INCLUDED_GR_BASIC_BLOCK_PYTHON_HPP
#define INCLUDED_GR_BASIC_BLOCK_PYTHON_HPP

// #pragma once

#include <gnuradio/basic_block.h>
#include <pmt/pmt.h>

void export_basic_block(py::module& m)
{
    using basic_block      = gr::basic_block;

    py::class_<basic_block, boost::shared_ptr<basic_block>>(m, "basic_block")
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

        ;
} 

#endif /* INCLUDED_UHD_USRP_MULTI_USRP_PYTHON_HPP */
