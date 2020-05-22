/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually edited  */
/* The following lines can be configured to regenerate this file during cmake      */
/* If manual edits are made, the following tags should be modified accordingly.    */
/* BINDTOOL_GEN_AUTOMATIC(0)                                                       */
/* BINDTOOL_USE_PYGCCXML(0)                                                        */
/* BINDTOOL_HEADER_FILE(adaptive_algorithm_cma.h) */
/* BINDTOOL_HEADER_FILE_HASH(4f720d6c8ec33fbe77c05861a5913926)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/digital/adaptive_algorithm_cma.h>
// pydoc.h is automatically generated in the build directory
#include <adaptive_algorithm_cma_pydoc.h>

void bind_adaptive_algorithm_cma(py::module& m)
{

    using adaptive_algorithm_cma = ::gr::digital::adaptive_algorithm_cma;


    py::class_<adaptive_algorithm_cma,
               gr::digital::adaptive_algorithm,
               std::shared_ptr<adaptive_algorithm_cma>>(
        m, "adaptive_algorithm_cma", D(adaptive_algorithm_cma))

        .def(py::init(&adaptive_algorithm_cma::make),
             py::arg("cons"),
             py::arg("step_size"),
             py::arg("modulus"),
             D(adaptive_algorithm_cma, make))


        .def("error",
             &adaptive_algorithm_cma::error,
             py::arg("out"),
             D(adaptive_algorithm_cma, error))


        .def("error_dd",
             &adaptive_algorithm_cma::error_dd,
             py::arg("u_n"),
             py::arg("decision"),
             D(adaptive_algorithm_cma, error_dd))


        .def("error_tr",
             &adaptive_algorithm_cma::error_tr,
             py::arg("u_n"),
             py::arg("d_n"),
             D(adaptive_algorithm_cma, error_tr))


        .def("update_taps",
             &adaptive_algorithm_cma::update_taps,
             py::arg("taps"),
             py::arg("in"),
             py::arg("error"),
             py::arg("decision"),
             py::arg("num_taps"),
             D(adaptive_algorithm_cma, update_taps))


        .def("update_tap",
             &adaptive_algorithm_cma::update_tap,
             py::arg("tap"),
             py::arg("u_n"),
             py::arg("err"),
             py::arg("decision"),
             D(adaptive_algorithm_cma, update_tap))

        ;
}
