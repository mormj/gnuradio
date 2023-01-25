/* -*- c++ -*- */
/*
 * Copyright 2023 Block Author
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/bench/copy.h>

namespace gr {
namespace bench {

template <class T>
class copy_cpu : public copy<T>
{
public:
    copy_cpu(const typename copy<T>::block_args& args);

    work_return_t work(work_io& wio) override;

private:
    bool _use_memcpy;
    size_t _nproduce;
};


} // namespace bench
} // namespace gr
