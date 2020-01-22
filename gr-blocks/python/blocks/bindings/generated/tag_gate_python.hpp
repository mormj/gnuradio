

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

#ifndef INCLUDED_GR_BLOCKS_TAG_GATE_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_TAG_GATE_PYTHON_HPP

#include <gnuradio/sync_block.h>
#include <gnuradio/blocks/tag_gate.h>

void bind_tag_gate(py::module& m)
{
    using tag_gate    = gr::blocks::tag_gate;


    py::class_<tag_gate,gr::sync_block,
        std::shared_ptr<tag_gate>>(m, "tag_gate")

        .def(py::init(&tag_gate::make),
           py::arg("item_size"), 
           py::arg("propagate_tags") = false 
        )
        

        .def("set_propagation",&tag_gate::set_propagation,
            py::arg("propagate_tags") 
        )
        .def("set_single_key",&tag_gate::set_single_key,
            py::arg("single_key") 
        )
        .def("single_key",&tag_gate::single_key)
        .def("to_basic_block",[](std::shared_ptr<tag_gate> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_BLOCKS_TAG_GATE_PYTHON_HPP */
