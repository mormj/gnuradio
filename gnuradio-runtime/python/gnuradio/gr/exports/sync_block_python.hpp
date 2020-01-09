//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_GR_SYNC_BLOCK_PYTHON_HPP
#define INCLUDED_GR_SYNC_BLOCK_PYTHON_HPP

// #pragma once

#include <gnuradio/sync_block.h>
#include <pmt/pmt.h>

void export_sync_block(py::module& m)
{
    using sync_block = gr::sync_block;
    py::class_<sync_block, gr::block, std::shared_ptr<sync_block>>(m, "sync_block")

        ;
}

#endif /* INCLUDED_GR_SYNC_BLOCK_PYTHON_HPP */
