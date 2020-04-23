/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/blocks/float_to_complex.h>
// pydoc.h is automatically generated in the build directory
#include <float_to_complex_pydoc.h>

void bind_float_to_complex(py::module& m)
{

    using float_to_complex = ::gr::blocks::float_to_complex;


    py::class_<float_to_complex,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<float_to_complex>>(
        m, "float_to_complex", D(float_to_complex))

        .def(py::init(&float_to_complex::make),
             py::arg("vlen") = 1,
             D(float_to_complex, make))


        ;
}
