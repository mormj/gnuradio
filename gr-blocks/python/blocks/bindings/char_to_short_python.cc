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

#include <gnuradio/blocks/char_to_short.h>
// pydoc.h is automatically generated in the build directory
#include <char_to_short_pydoc.h>

void bind_char_to_short(py::module& m)
{

    using char_to_short = ::gr::blocks::char_to_short;


    py::class_<char_to_short,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<char_to_short>>(m, "char_to_short", D(char_to_short))

        .def(py::init(&char_to_short::make), py::arg("vlen") = 1, D(char_to_short, make))


        ;
}
