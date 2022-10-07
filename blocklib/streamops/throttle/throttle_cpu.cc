/* -*- c++ -*- */
/*
 * Copyright 2005-2011 Free Software Foundation, Inc.
 * Copyright 2021 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "throttle_cpu.h"
#include "throttle_cpu_gen.h"
#include <thread>

namespace gr {
namespace streamops {

throttle_cpu::throttle_cpu(block_args args)
    : INHERITED_CONSTRUCTORS, d_ignore_tags(args.ignore_tags)
{
    set_sample_rate(args.samples_per_sec);
}

void throttle_cpu::set_sample_rate(double rate)
{
    // changing the sample rate performs a reset of state params
    d_start = std::chrono::steady_clock::now();
    d_total_samples = 0;
    d_sample_rate = rate;
    d_sample_period = std::chrono::duration<double>(1 / rate);
}

bool throttle_cpu::start()
{
    d_start = std::chrono::steady_clock::now();
    d_total_samples = 0;
    return block::start();
}

work_return_t throttle_cpu::work(work_io& wio)
{
    if (d_sleeping) {
        wio.produce_each(0);
        wio.consume_each(0);
        return work_return_t::OK;
    }

    auto in = wio.inputs()[0].items<uint8_t>();
    auto nitems = wio.inputs()[0].n_items;

    auto now = std::chrono::steady_clock::now();
    auto expected_time = d_start + d_sample_period * (d_total_samples + nitems);
    int n = nitems;
    if (expected_time > now) {
        if (pmtf::get_as<bool>(*param_trickle_mode)) {
            // produce as many samples as we can up to the current time
            int nsamps = (now - d_start) / d_sample_period - d_total_samples;
            if (nsamps > 0) {
                n = nsamps;
            }
            else {

                this->come_back_later(pmtf::get_as<int32_t>(*param_trickle_timer_ms));
                n = 0;
            }
        }
        else {

            this->come_back_later(
                std::chrono::duration_cast<std::chrono::milliseconds>(expected_time - now)
                    .count());
            n = 0;
        }
    }

    if (wio.outputs()[0].bufp()) {
        auto out = wio.outputs()[0].items<uint8_t>();
        if (n) {
            std::memcpy(out, in, n * wio.outputs()[0].buf().item_size());
        }
        wio.produce_each(n);
    }
    d_total_samples += n;
    wio.consume_each(n);

    d_debug_logger->debug("Throttle produced {}", n);
    return work_return_t::OK;
}


} // namespace streamops
} // namespace gr