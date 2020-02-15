/* -*- c++ -*- */
/*
 * Copyright 2011-2013,2017 Free Software Foundation, Inc.
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

#ifndef INCLUDED_RUNTIME_BLOCK_GATEWAY_H
#define INCLUDED_RUNTIME_BLOCK_GATEWAY_H

#include <gnuradio/api.h>
#include <gnuradio/block.h>

#include <pybind11/pybind11.h> // must be first
namespace py = pybind11;

namespace gr {

/*!
 * The gateway block which performs all the magic.
 *
 * The gateway provides access to all the gr::block routines.
 */
class GR_RUNTIME_API block_gateway : virtual public gr::block
{
private:
    py::handle d_py_handle;
public:
    // gr::block_gateway::sptr
    typedef std::shared_ptr<block_gateway> sptr;

    /*!
     * Make a new gateway block.
     * \param py_object the pybind11 object with callback
     * \param name the name of the block (Ex: "Shirley")
     * \param in_sig the input signature for this block
     * \param out_sig the output signature for this block
     * \return a new gateway block
     */
    static sptr make(const py::object& py_handle,
                     const std::string& name,
                     gr::io_signature::sptr in_sig,
                     gr::io_signature::sptr out_sig)
{
    return block_gateway::sptr(
        new block_gateway_impl(py_handle, name, in_sig, out_sig));
}

};

} /* namespace gr */

#endif /* INCLUDED_RUNTIME_BLOCK_GATEWAY_H */
