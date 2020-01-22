

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

#ifndef INCLUDED_GR_BLOCKS_PDU_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_PDU_PYTHON_HPP

#include <gnuradio/blocks/pdu.h>

void bind_pdu(py::module& m)
{

    py::enum_<gr::blocks::vector_type>(m,"vector_type")
        .value("byte_t", gr::blocks::byte_t) // 0
        .value("float_t", gr::blocks::float_t) // 1
        .value("complex_t", gr::blocks::complex_t) // 2
    ;

    m.def("pdu_port_id",&gr::blocks::pdu_port_id);
    m.def("itemsize",&gr::blocks::itemsize,
        py::arg("type") 
    );
    m.def("type_matches",&gr::blocks::type_matches,
        py::arg("type"), 
        py::arg("v") 
    );
    m.def("make_pdu_vector",&gr::blocks::make_pdu_vector,
        py::arg("type"), 
        py::arg("buf"), 
        py::arg("items") 
    );
    m.def("type_from_pmt",&gr::blocks::type_from_pmt,
        py::arg("vector") 
    );
} 

#endif /* INCLUDED_GR_BLOCKS_PDU_PYTHON_HPP */
