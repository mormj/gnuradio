

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_ANALOG_FREQUENCY_MODULATOR_FC_PYTHON_HPP
#define INCLUDED_GR_ANALOG_FREQUENCY_MODULATOR_FC_PYTHON_HPP

#include <gnuradio/analog/frequency_modulator_fc.h>

void bind_frequency_modulator_fc(py::module& m)
{
    using frequency_modulator_fc    = gr::analog::frequency_modulator_fc;


    py::class_<frequency_modulator_fc,gr::sync_block,
        std::shared_ptr<frequency_modulator_fc>>(m, "frequency_modulator_fc")

        .def(py::init(&frequency_modulator_fc::make),
           py::arg("sensitivity") 
        )
        

        .def("set_sensitivity",&frequency_modulator_fc::set_sensitivity,
            py::arg("sens") 
        )
        .def("sensitivity",&frequency_modulator_fc::sensitivity)
        .def("to_basic_block",[](std::shared_ptr<frequency_modulator_fc> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_ANALOG_FREQUENCY_MODULATOR_FC_PYTHON_HPP */
