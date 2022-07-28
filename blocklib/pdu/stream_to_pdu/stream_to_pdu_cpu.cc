/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "stream_to_pdu_cpu.h"
#include "stream_to_pdu_cpu_gen.h"

namespace gr {
namespace pdu {

template <class T>
stream_to_pdu_cpu<T>::stream_to_pdu_cpu(const typename stream_to_pdu<T>::block_args& args)
    : INHERITED_CONSTRUCTORS(T), d_packet_len(args.packet_len)
{
    this->set_output_multiple(d_packet_len);
}

template <class T>
work_return_t stream_to_pdu_cpu<T>::work(work_io& wio)
{
    auto n_pdu = wio.inputs()[0].n_items / d_packet_len;
    auto in = wio.inputs()[0].items<T>();

    for (size_t n = 0; n < n_pdu; n++) {
        auto pdu_out =
            pdu_wrap(get_data_type<T>(), (void *)(in + n * d_packet_len * itemsize), d_packet_len * itemsize);

        pdu_out["packet_len"] = d_packet_len;

        get_message_port("pdus")->post(pdu_out);
    }

    wio.consume_each(n_pdu * d_packet_len);
    return work_return_t::OK;
}

} /* namespace pdu */
} /* namespace gr */
