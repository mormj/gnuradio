

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

#ifndef INCLUDED_GR_BLOCKS_THRESHOLD_FF_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_THRESHOLD_FF_PYTHON_HPP

#include <gnuradio/blocks/threshold_ff.h>

void bind_threshold_ff(py::module& m)
{
    using threshold_ff    = gr::blocks::threshold_ff;


    py::class_<threshold_ff,  sync_block,
        std::shared_ptr<threshold_ff>>(m, "threshold_ff")

        .def(py::init(&threshold_ff::make),
           py::arg("lo"), 
           py::arg("hi"), 
           py::arg("initial_state") = 0 
        )
        

        .def("lo",&threshold_ff::lo)
        .def("set_lo",&threshold_ff::set_lo,
            py::arg("lo") 
        )
        .def("hi",&threshold_ff::hi)
        .def("set_hi",&threshold_ff::set_hi,
            py::arg("hi") 
        )
        .def("last_state",&threshold_ff::last_state)
        .def("set_last_state",&threshold_ff::set_last_state,
            py::arg("last_state") 
        )

        ;


} 

#endif /* INCLUDED_GR_BLOCKS_THRESHOLD_FF_PYTHON_HPP */
