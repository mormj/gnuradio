/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <queue>
#include <pmtf/wrap.hpp>

#include <gnuradio/pdu/pdu_to_stream.h>
#include <gnuradio/pdu.h>

namespace gr {
namespace pdu {

class pdu_to_stream_cpu : public pdu_to_stream
{
public:
    pdu_to_stream_cpu(const typename pdu_to_stream::block_args& args);
    
    virtual work_return_t work(work_io& wio) override;

private:
    data_type_t d_data_type;

    std::queue<pmtf::pmt> d_pmt_queue;
    pdu_wrap d_pdu;
    bool d_vec_ready = false;
    size_t d_vec_idx = 0;

    void handle_msg_pdus(pmtf::pmt msg) override;
};


} // namespace pdu
} // namespace gr
