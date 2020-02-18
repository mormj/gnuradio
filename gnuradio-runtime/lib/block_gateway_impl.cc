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
#include <pybind11/embed.h>

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


block_gateway::sptr block_gateway::make(const py::object& p,
                                        const std::string& name,
                                        gr::io_signature::sptr in_sig,
                                        gr::io_signature::sptr out_sig)
{
    return block_gateway::sptr(
        new block_gateway_impl(p, name, in_sig, out_sig));
}

block_gateway_impl::block_gateway_impl(const py::handle& p,
                                       const std::string& name,
                                       gr::io_signature::sptr in_sig,
                                       gr::io_signature::sptr out_sig)
    : block(name, in_sig, out_sig)//, _py_handle(py_object)
{
    _py_handle = p;
}

void block_gateway_impl::forecast(int noutput_items, gr_vector_int& ninput_items_required)
{
    py::gil_scoped_acquire acquire;
    // ninput_items_required[0] = noutput_items; // todo: replace with call back to python
    py::object ret_ninput_items_required = _py_handle.attr("handle_forecast")(noutput_items, ninput_items_required.size());

    gr_vector_int tmp_int_vector =  ret_ninput_items_required.cast<std::vector<int>>();
    ninput_items_required = tmp_int_vector;

    std::cout << noutput_items << "," << ninput_items_required[0] << std::endl;

}

int block_gateway_impl::general_work(int noutput_items,
                                     gr_vector_int& ninput_items,
                                     gr_vector_const_void_star& input_items,
                                     gr_vector_void_star& output_items)
{
    py::gil_scoped_acquire acquire;
    // py::scoped_interpreter guard{};

    // std::vector<uint64_t> input_item_ptrs, output_item_ptrs;

    // for (const void *p : input_items)
    // {
    //     input_item_ptrs.push_back(uint64_t(p));
    // }

    // for (void *p : output_items)
    // {
    //     output_item_ptrs.push_back(uint64_t(p));
    // }

    // std::vector<py::buffer_info> in_buffers;
    // int i=0;
    // for (const void *b : input_items)
    // {
    //     in_buffers.push_back(
    //         py::buffer_info(
    //             b,                            /* Pointer to buffer */
    //             sizeof(gr_complex),                       /* Size of one scalar */
    //             std::string("gr_complex"), //py::format_descriptor<gr_complex>::format(), /* Python struct-style format descriptor */
    //             1,                                   /* Number of dimensions */
    //             { ninput_items[i], },              /* Buffer dimensions */
    //             { sizeof(gr_complex), }
    //         )
    //     );
    //     i++;
    // }
    
    // _py_handle.attr("test")();
    std::cout << "c++ ninput_items = " << ninput_items[0] << std::endl;
    py::object ret = _py_handle.attr("general_work")(noutput_items, ninput_items, input_items, output_items);
    
    return ret.cast<int>();;
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
