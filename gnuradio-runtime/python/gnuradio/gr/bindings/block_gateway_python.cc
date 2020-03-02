/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/block_gateway.h>

void bind_block_gateway(py::module& m)
{
    using block_gateway    = gr::block_gateway;
    py::class_<block_gateway, gr::block, std::shared_ptr<block_gateway>>(m, "block_gateway")

        .def(py::init(&block_gateway::make),
            py::arg("p"),
            py::arg("name"),
            py::arg("in_sig"),
            py::arg("out_sig"))
        ;

    py::enum_<gr::gw_block_t>(m,"gw_block_t")
        .value("GW_BLOCK_GENERAL", gr::GW_BLOCK_GENERAL) 
        .value("GW_BLOCK_SYNC", gr::GW_BLOCK_SYNC) 
        .value("GW_BLOCK_DECIM", gr::GW_BLOCK_DECIM) 
        .value("GW_BLOCK_INTERP", gr::GW_BLOCK_INTERP) 
        .export_values()
    ;

}
