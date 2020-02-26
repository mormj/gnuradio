/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/dtv/dvbt_energy_dispersal.h>

void bind_dvbt_energy_dispersal(py::module& m)
{
    using dvbt_energy_dispersal    = gr::dtv::dvbt_energy_dispersal;


    py::class_<dvbt_energy_dispersal,gr::block,
        std::shared_ptr<dvbt_energy_dispersal>>(m, "dvbt_energy_dispersal")

        .def(py::init(&dvbt_energy_dispersal::make),
           py::arg("nsize") 
        )
        

        .def("to_basic_block",[](std::shared_ptr<dvbt_energy_dispersal> p){
            return p->to_basic_block();
        })
        ;


} 
