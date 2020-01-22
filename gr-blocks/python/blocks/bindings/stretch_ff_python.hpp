

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

#ifndef INCLUDED_GR_BLOCKS_STRETCH_FF_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_STRETCH_FF_PYTHON_HPP

#include <gnuradio/blocks/stretch_ff.h>

void bind_stretch_ff(py::module& m)
{
    using stretch_ff    = gr::blocks::stretch_ff;


    py::class_<stretch_ff,  sync_block,
        std::shared_ptr<stretch_ff>>(m, "stretch_ff")

        .def(py::init(&stretch_ff::make),
           py::arg("lo"), 
           py::arg("vlen") = 1 
        )
        

        .def("lo",&stretch_ff::lo)
        .def("set_lo",&stretch_ff::set_lo,
            py::arg("lo") 
        )
        .def("vlen",&stretch_ff::vlen)

        ;


} 

#endif /* INCLUDED_GR_BLOCKS_STRETCH_FF_PYTHON_HPP */
