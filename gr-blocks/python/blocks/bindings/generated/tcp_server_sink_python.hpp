

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


/* This file is automatically generated using the gen_nonblock_bindings.py tool */

#ifndef INCLUDED_GR_BLOCKS_TCP_SERVER_SINK_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_TCP_SERVER_SINK_PYTHON_HPP

#include <gnuradio/sync_block.h>
#include <gnuradio/blocks/tcp_server_sink.h>

void bind_tcp_server_sink(py::module& m)
{
    using tcp_server_sink    = gr::blocks::tcp_server_sink;


    py::class_<tcp_server_sink,gr::sync_block,
        std::shared_ptr<tcp_server_sink>>(m, "tcp_server_sink")

        .def(py::init(&tcp_server_sink::make),
           py::arg("itemsize"), 
           py::arg("host"), 
           py::arg("port"), 
           py::arg("noblock") = false 
        )
        

        .def("to_basic_block",[](std::shared_ptr<tcp_server_sink> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_BLOCKS_TCP_SERVER_SINK_PYTHON_HPP */
