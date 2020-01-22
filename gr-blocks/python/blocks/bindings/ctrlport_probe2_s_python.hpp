

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

#ifndef INCLUDED_GR_BLOCKS_CTRLPORT_PROBE2_S_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_CTRLPORT_PROBE2_S_PYTHON_HPP

#include <gnuradio/blocks/ctrlport_probe2_s.h>

void bind_ctrlport_probe2_s(py::module& m)
{
    using ctrlport_probe2_s    = gr::blocks::ctrlport_probe2_s;


    py::class_<ctrlport_probe2_s,  sync_block,
        std::shared_ptr<ctrlport_probe2_s>>(m, "ctrlport_probe2_s")

        .def(py::init(&ctrlport_probe2_s::make),
           py::arg("id"), 
           py::arg("desc"), 
           py::arg("len"), 
           py::arg("disp_mask") 
        )
        

        .def("get",&ctrlport_probe2_s::get)
        .def("set_length",&ctrlport_probe2_s::set_length,
            py::arg("len") 
        )

        ;


} 

#endif /* INCLUDED_GR_BLOCKS_CTRLPORT_PROBE2_S_PYTHON_HPP */
