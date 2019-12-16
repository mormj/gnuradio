//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_GR_HIER_BLOCK2_PYTHON_HPP
#define INCLUDED_GR_HIER_BLOCK2_PYTHON_HPP

// #pragma once

#include <gnuradio/hier_block2.h>
#include <gnuradio/basic_block.h>

void export_hier_block2(py::module& m)
{
    using hier_block2      = gr::hier_block2;
    using basic_block_sptr = boost::shared_ptr<gr::basic_block>;

    py::class_<hier_block2, boost::shared_ptr<hier_block2>>(m, "hier_block2")     
        
        .def(py::init(&gr::make_hier_block2))
        .def("connect", (void (hier_block2::*)(basic_block_sptr)) &hier_block2::connect)
        .def("connect", (void (hier_block2::*)(basic_block_sptr, int, basic_block_sptr, int)) &hier_block2::connect)
        
        // .def("asdf", (void (hier_block2::*)()) &hier_block2::asdf)
        // .def("asdf", (void (hier_block2::*)(int)) &hier_block2::asdf)
        ;
} 

#endif /* INCLUDED_UHD_USRP_MULTI_USRP_PYTHON_HPP */
