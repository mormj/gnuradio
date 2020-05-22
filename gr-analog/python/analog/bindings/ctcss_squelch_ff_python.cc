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
/* BINDTOOL_HEADER_FILE(ctcss_squelch_ff.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(e3921bc97f0f8813b708306467a41d49)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/analog/ctcss_squelch_ff.h>
// pydoc.h is automatically generated in the build directory
#include <ctcss_squelch_ff_pydoc.h>

void bind_ctcss_squelch_ff(py::module& m)
{

    using ctcss_squelch_ff = ::gr::analog::ctcss_squelch_ff;


    py::class_<ctcss_squelch_ff,
               gr::analog::squelch_base_ff,
               std::shared_ptr<ctcss_squelch_ff>>(
        m, "ctcss_squelch_ff", D(ctcss_squelch_ff))

        .def(py::init(&ctcss_squelch_ff::make),
             py::arg("rate"),
             py::arg("freq"),
             py::arg("level"),
             py::arg("len"),
             py::arg("ramp"),
             py::arg("gate"),
             D(ctcss_squelch_ff, make))


        .def("squelch_range",
             &ctcss_squelch_ff::squelch_range,
             D(ctcss_squelch_ff, squelch_range))


        .def("level", &ctcss_squelch_ff::level, D(ctcss_squelch_ff, level))


        .def("set_level",
             &ctcss_squelch_ff::set_level,
             py::arg("level"),
             D(ctcss_squelch_ff, set_level))


        .def("len", &ctcss_squelch_ff::len, D(ctcss_squelch_ff, len))


        .def("frequency", &ctcss_squelch_ff::frequency, D(ctcss_squelch_ff, frequency))


        .def("set_frequency",
             &ctcss_squelch_ff::set_frequency,
             py::arg("frequency"),
             D(ctcss_squelch_ff, set_frequency))


        .def("ramp", &ctcss_squelch_ff::ramp, D(ctcss_squelch_ff, ramp))


        .def("set_ramp",
             &ctcss_squelch_ff::set_ramp,
             py::arg("ramp"),
             D(ctcss_squelch_ff, set_ramp))


        .def("gate", &ctcss_squelch_ff::gate, D(ctcss_squelch_ff, gate))


        .def("set_gate",
             &ctcss_squelch_ff::set_gate,
             py::arg("gate"),
             D(ctcss_squelch_ff, set_gate))


        .def("unmuted", &ctcss_squelch_ff::unmuted, D(ctcss_squelch_ff, unmuted))

        ;
}
