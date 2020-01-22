

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

#ifndef INCLUDED_GR_BLOCKS_CHAR_TO_FLOAT_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_CHAR_TO_FLOAT_PYTHON_HPP

#include <gnuradio/sync_block.h>
#include <gnuradio/blocks/char_to_float.h>

void bind_char_to_float(py::module& m)
{
    using char_to_float    = gr::blocks::char_to_float;


    py::class_<char_to_float,gr::sync_block,
        std::shared_ptr<char_to_float>>(m, "char_to_float")

        .def(py::init(&char_to_float::make),
           py::arg("vlen") = 1, 
           py::arg("scale") = 1. 
        )
        

        .def("scale",&char_to_float::scale)
        .def("set_scale",&char_to_float::set_scale,
            py::arg("scale") 
        )
        .def("to_basic_block",[](std::shared_ptr<char_to_float> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_BLOCKS_CHAR_TO_FLOAT_PYTHON_HPP */
