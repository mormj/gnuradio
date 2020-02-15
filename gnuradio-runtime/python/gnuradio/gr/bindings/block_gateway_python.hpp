

/* Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_GR_BLOCK_GATEWAY_PYTHON_HPP
#define INCLUDED_GR_BLOCK_GATEWAY_PYTHON_HPP

#include <gnuradio/block_gateway.h>

void bind_block_gateway(py::module& m)
{
    using block_gateway    = gr::block_gateway;
    using block_gw_message_type = gr::block_gw_message_type;

    py::class_<block_gateway, gr::block, std::shared_ptr<block_gateway>>(m, "block_gateway")

        .def(py::init(&block_gateway::make))
} 

#endif /* INCLUDED_GR_BLOCK_GATEWAY_PYTHON_HPP */
