/* -*- c++ -*- */
/*
 * Copyright 2023 Block Author
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "copy_cpu.h"
#include "copy_cpu_gen.h"

namespace gr {
namespace bench {

template <class T>
copy_cpu<T>::copy_cpu(const typename copy<T>::block_args& args)
    : INHERITED_CONSTRUCTORS(T), _use_memcpy(args.use_memcpy), _nproduce(args.nproduce)
{
}

template <class T>
work_return_t copy_cpu<T>::work(work_io& wio)
{
    int n;

    auto noutput_items = wio.outputs()[0].n_items;

    if (_nproduce) {
        n = _nproduce;
    }
    else {
        n = noutput_items;
    }
    auto* in = wio.inputs()[0].items<T>();
    auto* out = wio.outputs()[0].items<T>();

    if (_use_memcpy) {
        memcpy(out, in, n * sizeof(T));
    }
    else {
        for (size_t ii = 0; ii < n; ii++) {
            out[ii] = in[ii];
        }
    }

    wio.produce_each(n);
    return work_return_t::OK;
}

} /* namespace bench */
} /* namespace gr */
