

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

#ifndef INCLUDED_GR_BLOCKS_TAGGED_STREAM_MUX_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_TAGGED_STREAM_MUX_PYTHON_HPP

#include <gnuradio/blocks/tagged_stream_mux.h>

void bind_tagged_stream_mux(py::module& m)
{
    using tagged_stream_mux    = gr::blocks::tagged_stream_mux;


    py::class_<tagged_stream_mux,  tagged_stream_block,
        std::shared_ptr<tagged_stream_mux>>(m, "tagged_stream_mux")

        .def(py::init(&tagged_stream_mux::make),
           py::arg("itemsize"), 
           py::arg("lengthtagname"), 
           py::arg("tag_preserve_head_pos") = 0 
        )
        


        ;


} 

#endif /* INCLUDED_GR_BLOCKS_TAGGED_STREAM_MUX_PYTHON_HPP */
