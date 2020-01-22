

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

#ifndef INCLUDED_GR_BLOCKS_KEEP_ONE_IN_N_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_KEEP_ONE_IN_N_PYTHON_HPP

#include <gnuradio/block.h>
#include <gnuradio/blocks/keep_one_in_n.h>

void bind_keep_one_in_n(py::module& m)
{
    using keep_one_in_n    = gr::blocks::keep_one_in_n;


    py::class_<keep_one_in_n,gr::block,
        std::shared_ptr<keep_one_in_n>>(m, "keep_one_in_n")

        .def(py::init(&keep_one_in_n::make),
           py::arg("itemsize"), 
           py::arg("n") 
        )
        

        .def("set_n",&keep_one_in_n::set_n,
            py::arg("n") 
        )
        .def("to_basic_block",[](std::shared_ptr<keep_one_in_n> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_BLOCKS_KEEP_ONE_IN_N_PYTHON_HPP */
