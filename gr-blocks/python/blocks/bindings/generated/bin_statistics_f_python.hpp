

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

#ifndef INCLUDED_GR_BLOCKS_BIN_STATISTICS_F_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_BIN_STATISTICS_F_PYTHON_HPP

#include <gnuradio/sync_block.h>
#include <gnuradio/blocks/bin_statistics_f.h>

void bind_bin_statistics_f(py::module& m)
{
    using bin_statistics_f    = gr::blocks::bin_statistics_f;


    py::class_<bin_statistics_f,gr::sync_block,
        std::shared_ptr<bin_statistics_f>>(m, "bin_statistics_f")

        .def(py::init(&bin_statistics_f::make),
           py::arg("vlen"), 
           py::arg("msgq"), 
           py::arg("tune"), 
           py::arg("tune_delay"), 
           py::arg("dwell_delay") 
        )
        

        .def("to_basic_block",[](std::shared_ptr<bin_statistics_f> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_BLOCKS_BIN_STATISTICS_F_PYTHON_HPP */
