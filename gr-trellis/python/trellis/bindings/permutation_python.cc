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

#include <gnuradio/trellis/permutation.h>

void bind_permutation(py::module& m)
{
    using permutation    = gr::trellis::permutation;


    py::class_<permutation,gr::sync_block,
        std::shared_ptr<permutation>>(m, "permutation")

        .def(py::init(&permutation::make),
           py::arg("K"), 
           py::arg("TABLE"), 
           py::arg("SYMS_PER_BLOCK"), 
           py::arg("NBYTES") 
        )
        

        .def("K",&permutation::K)
        .def("TABLE",&permutation::TABLE)
        .def("SYMS_PER_BLOCK",&permutation::SYMS_PER_BLOCK)
        .def("BYTES_PER_SYMBOL",&permutation::BYTES_PER_SYMBOL)
        .def("set_K",&permutation::set_K,
            py::arg("K") 
        )
        .def("set_TABLE",&permutation::set_TABLE,
            py::arg("table") 
        )
        .def("set_SYMS_PER_BLOCK",&permutation::set_SYMS_PER_BLOCK,
            py::arg("spb") 
        )
        .def("to_basic_block",[](std::shared_ptr<permutation> p){
            return p->to_basic_block();
        })
        ;


} 
