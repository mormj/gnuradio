

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

#ifndef INCLUDED_GR_BLOCKS_MULTIPLY_BY_TAG_VALUE_CC_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_MULTIPLY_BY_TAG_VALUE_CC_PYTHON_HPP

#include <gnuradio/sync_block.h>
#include <gnuradio/blocks/multiply_by_tag_value_cc.h>

void bind_multiply_by_tag_value_cc(py::module& m)
{
    using multiply_by_tag_value_cc    = gr::blocks::multiply_by_tag_value_cc;


    py::class_<multiply_by_tag_value_cc,gr::sync_block,
        std::shared_ptr<multiply_by_tag_value_cc>>(m, "multiply_by_tag_value_cc")

        .def(py::init(&multiply_by_tag_value_cc::make),
           py::arg("tag_name"), 
           py::arg("vlen") = 1 
        )
        

        .def("k",&multiply_by_tag_value_cc::k)
        .def("to_basic_block",[](std::shared_ptr<multiply_by_tag_value_cc> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_BLOCKS_MULTIPLY_BY_TAG_VALUE_CC_PYTHON_HPP */
