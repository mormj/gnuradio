

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_ANALOG_PWR_SQUELCH_FF_PYTHON_HPP
#define INCLUDED_GR_ANALOG_PWR_SQUELCH_FF_PYTHON_HPP

#include <gnuradio/analog/pwr_squelch_ff.h>

void bind_pwr_squelch_ff(py::module& m)
{
    using pwr_squelch_ff    = gr::analog::pwr_squelch_ff;


    py::class_<pwr_squelch_ff,gr::analog::squelch_base_ff,
        std::shared_ptr<pwr_squelch_ff>>(m, "pwr_squelch_ff")

        .def(py::init(&pwr_squelch_ff::make),
           py::arg("db"), 
           py::arg("alpha") = 1.0E-4, 
           py::arg("ramp") = 0, 
           py::arg("gate") = false 
        )
        

        .def("squelch_range",&pwr_squelch_ff::squelch_range)
        .def("threshold",&pwr_squelch_ff::threshold)
        .def("set_threshold",&pwr_squelch_ff::set_threshold,
            py::arg("db") 
        )
        .def("set_alpha",&pwr_squelch_ff::set_alpha,
            py::arg("alpha") 
        )
        .def("ramp",&pwr_squelch_ff::ramp)
        .def("set_ramp",&pwr_squelch_ff::set_ramp,
            py::arg("ramp") 
        )
        .def("gate",&pwr_squelch_ff::gate)
        .def("set_gate",&pwr_squelch_ff::set_gate,
            py::arg("gate") 
        )
        .def("unmuted",&pwr_squelch_ff::unmuted)
        .def("to_basic_block",[](std::shared_ptr<pwr_squelch_ff> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_ANALOG_PWR_SQUELCH_FF_PYTHON_HPP */
