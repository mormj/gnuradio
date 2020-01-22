

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

#ifndef INCLUDED_GR_BLOCKS_ADD_CONST_SS_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_ADD_CONST_SS_PYTHON_HPP

#include <gnuradio/blocks/add_const_ss.h>

void bind_add_const_ss(py::module& m)
{
    using add_const_ss    = gr::blocks::add_const_ss;


    py::class_<add_const_ss,  sync_block,
        std::shared_ptr<add_const_ss>>(m, "add_const_ss")

        .def(py::init(&add_const_ss::make),
           py::arg("k") 
        )
        

        .def("k",&add_const_ss::k)
        .def("set_k",&add_const_ss::set_k,
            py::arg("k") 
        )

        ;


} 

#endif /* INCLUDED_GR_BLOCKS_ADD_CONST_SS_PYTHON_HPP */
