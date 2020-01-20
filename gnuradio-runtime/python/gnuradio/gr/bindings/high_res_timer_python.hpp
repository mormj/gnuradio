

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


#ifndef INCLUDED_GR_HIGH_RES_TIMER_PYTHON_HPP
#define INCLUDED_GR_HIGH_RES_TIMER_PYTHON_HPP

#include <gnuradio/high_res_timer.h>

void bind_high_res_timer(py::module& m)
{
    m.def("high_res_timer_now",&gr::high_res_timer_now);
    m.def("high_res_timer_now_perfmon",&gr::high_res_timer_now_perfmon);
    m.def("high_res_timer_tps",&gr::high_res_timer_tps);
    m.def("high_res_timer_epoch",&gr::high_res_timer_epoch);
} 

#endif /* INCLUDED_GR_HIGH_RES_TIMER_PYTHON_HPP */
