/*
 * Copyright 2011-2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#include "block_gateway_impl.h"
#include <gnuradio/io_signature.h>
#include <iostream>

namespace gr {

/***********************************************************************
 * Helper routines
 **********************************************************************/
template <typename OUT_T, typename IN_T>
void copy_pointers(OUT_T& out, const IN_T& in)
{
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        out[i] = (void*)(in[i]);
    }
}


block_gateway::sptr block_gateway::make(const py::object& py_object,
                                        const std::string& name,
                                        gr::io_signature::sptr in_sig,
                                        gr::io_signature::sptr out_sig)
{
    return block_gateway::sptr(
        new block_gateway_impl(py_object, name, in_sig, out_sig));
}

block_gateway_impl::block_gateway_impl(const py::object& py_object,
                                       const std::string& name,
                                       gr::io_signature::sptr in_sig,
                                       gr::io_signature::sptr out_sig)
    : block(name, in_sig, out_sig), _py_handle(py_object)
{
    
}

void block_gateway_impl::forecast(int noutput_items, gr_vector_int& ninput_items_required)
{

}

int block_gateway_impl::general_work(int noutput_items,
                                     gr_vector_int& ninput_items,
                                     gr_vector_const_void_star& input_items,
                                     gr_vector_void_star& output_items)
{
    return _py_handle.attr("general_work")(noutput_items, ninput_items, input_items, output_items);
}

int block_gateway_impl::work(int noutput_items,
                             gr_vector_const_void_star& input_items,
                             gr_vector_void_star& output_items)
{

}

int block_gateway_impl::fixed_rate_noutput_to_ninput(int noutput_items)
{
    
}

int block_gateway_impl::fixed_rate_ninput_to_noutput(int ninput_items)
{
    
}

bool block_gateway_impl::start(void)
{

}

bool block_gateway_impl::stop(void)
{

}

} /* namespace gr */
