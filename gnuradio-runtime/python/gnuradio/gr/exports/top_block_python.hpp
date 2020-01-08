//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_GR_TOP_BLOCK_PYTHON_HPP
#define INCLUDED_GR_TOP_BLOCK_PYTHON_HPP

// #pragma once

#include <gnuradio/top_block.h>
#include <gnuradio/runtime_types.h>

#define GR_PYTHON_BLOCKING_CODE(code) {                                 \
    PyThreadState *_save;                                               \
    _save = PyEval_SaveThread();                                        \
    try{code}                                                           \
    catch(...){PyEval_RestoreThread(_save); throw;}                     \
    PyEval_RestoreThread(_save);                                        \
}


void top_block_run_unlocked(gr::top_block_sptr r) noexcept(false)
{
    GR_PYTHON_BLOCKING_CODE
    (
        r->run();
    )
}

void top_block_start_unlocked(gr::top_block_sptr r, int max_noutput_items) noexcept(false)
{
    GR_PYTHON_BLOCKING_CODE
    (
        r->start(max_noutput_items);
    )
}

void top_block_wait_unlocked(gr::top_block_sptr r) noexcept(false)
{
    GR_PYTHON_BLOCKING_CODE
    (
        r->wait();
    )
}

void top_block_stop_unlocked(gr::top_block_sptr r) noexcept(false)
{
    GR_PYTHON_BLOCKING_CODE
    (
        r->stop();
    )
}

void top_block_unlock_unlocked(gr::top_block_sptr r) noexcept(false)
{
    GR_PYTHON_BLOCKING_CODE
    (
        r->unlock();
    )
}

void export_top_block(py::module& m)
{
    using top_block      = gr::top_block;

    py::class_<top_block, std::shared_ptr<top_block>>(m, "top_block_pb")
    // py::class_<top_block, gr::top_block_sptr>(m, "top_block_pb")
        // .def(py::init())
        .def(py::init(&gr::make_top_block))
        .def("start", &top_block::start)
        .def("stop", &top_block::stop)
        .def("run", &top_block::run)
        .def("wait", &top_block::wait)
        .def("unlock", &top_block::unlock)
        ;

    m.def("make_top_block", &gr::make_top_block);

    m.def("top_block_run_unlocked", &top_block_run_unlocked);
    m.def("top_block_start_unlocked", &top_block_start_unlocked);
    m.def("top_block_wait_unlocked", &top_block_wait_unlocked);
    m.def("top_block_stop_unlocked", &top_block_stop_unlocked);
    m.def("top_block_unlock_unlocked", &top_block_unlock_unlocked);
} 

#endif /* INCLUDED_UHD_USRP_MULTI_USRP_PYTHON_HPP */
