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

#include <gnuradio/random.h>
// pydoc.h is automatically generated in the build directory
#include <random_pydoc.h>

void bind_random(py::module& m)
{

    using random = ::gr::random;


    py::class_<random, std::shared_ptr<random>>(m, "random", D(random))

        .def(py::init<unsigned int, int, int>(),
             py::arg("seed") = 0,
             py::arg("min_integer") = 0,
             py::arg("max_integer") = 2,
             D(random, random, 0))
        .def(py::init<gr::random const&>(), py::arg("arg0"), D(random, random, 1))


        .def("reseed", &random::reseed, py::arg("seed"), D(random, reseed))


        .def("set_integer_limits",
             &random::set_integer_limits,
             py::arg("minimum"),
             py::arg("maximum"),
             D(random, set_integer_limits))


        .def("ran_int", &random::ran_int, D(random, ran_int))


        .def("ran1", &random::ran1, D(random, ran1))


        .def("gasdev", &random::gasdev, D(random, gasdev))


        .def("laplacian", &random::laplacian, D(random, laplacian))


        .def("rayleigh", &random::rayleigh, D(random, rayleigh))


        .def("impulse", &random::impulse, py::arg("factor"), D(random, impulse))


        .def("rayleigh_complex", &random::rayleigh_complex, D(random, rayleigh_complex))

        ;
}
