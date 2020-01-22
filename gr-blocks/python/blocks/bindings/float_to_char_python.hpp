

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

#ifndef INCLUDED_GR_BLOCKS_FLOAT_TO_CHAR_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_FLOAT_TO_CHAR_PYTHON_HPP

#include <gnuradio/blocks/float_to_char.h>

void bind_float_to_char(py::module& m)
{
    using float_to_char    = gr::blocks::float_to_char;


    py::class_<float_to_char,  sync_block,
        std::shared_ptr<float_to_char>>(m, "float_to_char")

        .def(py::init(&float_to_char::make),
           py::arg("vlen") = 1, 
           py::arg("scale") = 1. 
        )
        

        .def("scale",&float_to_char::scale)
        .def("set_scale",&float_to_char::set_scale,
            py::arg("scale") 
        )

        ;


} 

#endif /* INCLUDED_GR_BLOCKS_FLOAT_TO_CHAR_PYTHON_HPP */
