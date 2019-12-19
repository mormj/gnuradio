//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_BLOCKS_NULL_SINK_PYTHON_HPP
#define INCLUDED_BLOCKS_NULL_SINK_PYTHON_HPP

// #pragma once

#include <gnuradio/blocks/null_sink.h>
#include <gnuradio/sync_block.h>


void export_null_sink(py::module& m)
{
    using null_sink      = gr::blocks::null_sink;

    py::class_<null_sink, gr::sync_block, boost::shared_ptr<null_sink>>(m, "null_sink")
        .def(py::init(&null_sink::make))
        ;
} 


#endif /* INCLUDED_BLOCKS_null_sink_PYTHON_HPP */
