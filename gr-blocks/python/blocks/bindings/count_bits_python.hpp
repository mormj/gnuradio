

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

#ifndef INCLUDED_GR_BLOCKS_COUNT_BITS_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_COUNT_BITS_PYTHON_HPP

#include <gnuradio/blocks/count_bits.h>

void bind_count_bits(py::module& m)
{


    m.def("count_bits8",&gr::blocks::count_bits8,
        py::arg("x") 
    );
    m.def("count_bits16",&gr::blocks::count_bits16,
        py::arg("x") 
    );
    m.def("count_bits32",&gr::blocks::count_bits32,
        py::arg("x") 
    );
    m.def("count_bits64",&gr::blocks::count_bits64,
        py::arg("x") 
    );
} 

#endif /* INCLUDED_GR_BLOCKS_COUNT_BITS_PYTHON_HPP */
