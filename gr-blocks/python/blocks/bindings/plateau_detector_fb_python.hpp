

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

#ifndef INCLUDED_GR_BLOCKS_PLATEAU_DETECTOR_FB_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_PLATEAU_DETECTOR_FB_PYTHON_HPP

#include <gnuradio/blocks/plateau_detector_fb.h>

void bind_plateau_detector_fb(py::module& m)
{
    using plateau_detector_fb    = gr::blocks::plateau_detector_fb;


    py::class_<plateau_detector_fb,  block,
        std::shared_ptr<plateau_detector_fb>>(m, "plateau_detector_fb")

        .def(py::init(&plateau_detector_fb::make),
           py::arg("max_len"), 
           py::arg("threshold") = 0.90000000000000002 
        )
        

        .def("set_threshold",&plateau_detector_fb::set_threshold,
            py::arg("threshold") 
        )
        .def("threshold",&plateau_detector_fb::threshold)

        ;


} 

#endif /* INCLUDED_GR_BLOCKS_PLATEAU_DETECTOR_FB_PYTHON_HPP */
