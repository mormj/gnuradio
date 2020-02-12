

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

#ifndef INCLUDED_GR_SYNC_BLOCK_PYTHON_HPP
#define INCLUDED_GR_SYNC_BLOCK_PYTHON_HPP

#include <gnuradio/sync_block.h>

void bind_sync_block(py::module& m)
{
    using sync_block    = gr::sync_block;


    py::class_<sync_block, gr::block, std::shared_ptr<sync_block>>(m, "sync_block")
        .def("work",&sync_block::work,
            py::arg("noutput_items"), 
            py::arg("input_items"), 
            py::arg("output_items") 
        )
        .def("forecast",&sync_block::forecast,
            py::arg("noutput_items"), 
            py::arg("ninput_items_required") 
        )
        .def("general_work",&sync_block::general_work,
            py::arg("noutput_items"), 
            py::arg("ninput_items"), 
            py::arg("input_items"), 
            py::arg("output_items") 
        )
        .def("fixed_rate_ninput_to_noutput",&sync_block::fixed_rate_ninput_to_noutput,
            py::arg("ninput") 
        )
        .def("fixed_rate_noutput_to_ninput",&sync_block::fixed_rate_noutput_to_ninput,
            py::arg("noutput") 
        )
        ;

} 

#endif /* INCLUDED_GR_SYNC_BLOCK_PYTHON_HPP */
