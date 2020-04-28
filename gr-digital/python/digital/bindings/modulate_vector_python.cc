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

#include <gnuradio/basic_block.h>
#include <gnuradio/runtime_types.h>
#include <gnuradio/digital/modulate_vector.h>

void bind_modulate_vector(py::module& m)
{

    m.def("modulate_vector_bc",&::gr::digital::modulate_vector_bc,
      py::arg("modulator"),
      py::arg("data"),
      py::arg("taps"));
}
