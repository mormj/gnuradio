//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_GR_SYNC_DECIMATOR_PYTHON_HPP
#define INCLUDED_GR_SYNC_DECIMATOR_PYTHON_HPP

// #pragma once

#include <gnuradio/sync_decimator.h>
#include <pmt/pmt.h>

void bind_sync_decimator(py::module& m)
{
    using sync_decimator      = gr::sync_decimator;
    py::class_<sync_decimator, gr::block, std::shared_ptr<sync_decimator>>(m, "sync_decimator")

        ;
} 

#endif /* INCLUDED_GR_SYNC_DECIMATOR_PYTHON_HPP */
