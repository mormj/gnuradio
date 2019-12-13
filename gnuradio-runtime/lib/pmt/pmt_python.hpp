//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_PMT_PYTHON_HPP
#define INCLUDED_PMT_PYTHON_HPP

// #pragma once

#include <pmt/pmt.h>

int add(int i, int j) {
    return i + j;
}

void export_pmt(py::module& m)
{
    // using pmt_base      = pmt::pmt_base;
    // using pmt_t         = pmt::pmt_t;
    // py::class_<pmt_base, boost::shared_ptr<pmt_base>>(m, "pmt_t")
        // .def(py::init<const std::string&, const std::string&>())
        // ;
    // py::class_<pmt_t>(m, "pmt_t")
        // ;
    // py::class_<pmt::exception>(m,"exception");

    // m.def("is_bool", &pmt::is_bool);
    // m.def("to_bool", &pmt::to_bool);
    // m.def("add", &add, "A function which adds two numbers");
    m.def("asdf", &pmt::asdf, "A function which adds two numbers");

} 

#endif /* INCLUDED_UHD_USRP_MULTI_USRP_PYTHON_HPP */
