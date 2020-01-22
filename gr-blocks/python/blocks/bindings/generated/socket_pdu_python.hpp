

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

#ifndef INCLUDED_GR_BLOCKS_SOCKET_PDU_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_SOCKET_PDU_PYTHON_HPP

#include <gnuradio/block.h>
#include <gnuradio/blocks/socket_pdu.h>

void bind_socket_pdu(py::module& m)
{
    using socket_pdu    = gr::blocks::socket_pdu;


    py::class_<socket_pdu,gr::block,
        std::shared_ptr<socket_pdu>>(m, "socket_pdu")

        .def(py::init(&socket_pdu::make),
           py::arg("type"), 
           py::arg("addr"), 
           py::arg("port"), 
           py::arg("MTU") = 10000, 
           py::arg("tcp_no_delay") = false 
        )
        

        .def("to_basic_block",[](std::shared_ptr<socket_pdu> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_BLOCKS_SOCKET_PDU_PYTHON_HPP */
