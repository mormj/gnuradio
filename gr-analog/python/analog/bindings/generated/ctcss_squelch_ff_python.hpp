

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_ANALOG_CTCSS_SQUELCH_FF_PYTHON_HPP
#define INCLUDED_GR_ANALOG_CTCSS_SQUELCH_FF_PYTHON_HPP

#include <gnuradio/analog/ctcss_squelch_ff.h>

void bind_ctcss_squelch_ff(py::module& m)
{
    using ctcss_squelch_ff    = gr::analog::ctcss_squelch_ff;


    py::class_<ctcss_squelch_ff, gr::analog::squelch_base_ff,
        std::shared_ptr<ctcss_squelch_ff>>(m, "ctcss_squelch_ff")

        .def(py::init(&ctcss_squelch_ff::make),
           py::arg("rate"), 
           py::arg("freq"), 
           py::arg("level"), 
           py::arg("len"), 
           py::arg("ramp"), 
           py::arg("gate") 
        )
        

        .def("squelch_range",&ctcss_squelch_ff::squelch_range)
        .def("level",&ctcss_squelch_ff::level)
        .def("set_level",&ctcss_squelch_ff::set_level,
            py::arg("level") 
        )
        .def("len",&ctcss_squelch_ff::len)
        .def("frequency",&ctcss_squelch_ff::frequency)
        .def("set_frequency",&ctcss_squelch_ff::set_frequency,
            py::arg("frequency") 
        )
        .def("ramp",&ctcss_squelch_ff::ramp)
        .def("set_ramp",&ctcss_squelch_ff::set_ramp,
            py::arg("ramp") 
        )
        .def("gate",&ctcss_squelch_ff::gate)
        .def("set_gate",&ctcss_squelch_ff::set_gate,
            py::arg("gate") 
        )
        .def("unmuted",&ctcss_squelch_ff::unmuted)
        .def("to_basic_block",[](std::shared_ptr<ctcss_squelch_ff> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_ANALOG_CTCSS_SQUELCH_FF_PYTHON_HPP */
