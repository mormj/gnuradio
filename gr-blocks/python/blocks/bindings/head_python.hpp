

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


#ifndef INCLUDED_BLOCKS_HEAD_PYTHON_HPP
#define INCLUDED_BLOCKS_HEAD_PYTHON_HPP

#include <gnuradio/sync_block.h>
#include <gnuradio/blocks/head.h>

void bind_head(py::module& m)
{
    using head    = gr::blocks::head;

    py::class_<head, gr::sync_block, std::shared_ptr<head>>(m, "head")
        .def(py::init(&head::make),
            py::arg("sizeof_stream_item"), 
            py::arg("nitems") 
        )

        .def("set_length",&head::set_length)
        .def("to_basic_block",[](std::shared_ptr<head> p){
            return p->to_basic_block();
        })
        ;
} 

#endif /* INCLUDED_BLOCKS_HEAD_PYTHON_HPP */
