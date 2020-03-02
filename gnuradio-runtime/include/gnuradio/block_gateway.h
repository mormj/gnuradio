/* -*- c++ -*- */
/*
 * Copyright 2011-2013,2017,2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_RUNTIME_BLOCK_GATEWAY_H
#define INCLUDED_RUNTIME_BLOCK_GATEWAY_H

#include <gnuradio/api.h>
#include <gnuradio/block.h>

#include <pybind11/pybind11.h> // must be first
#include <pybind11/stl.h>
namespace py = pybind11;

namespace gr {

/*!
 * The gateway block which performs all the magic.
 *
 * The gateway provides access to all the gr::block routines.
 */
typedef enum {
    GW_BLOCK_GENERAL = 0,
    GW_BLOCK_SYNC,
    GW_BLOCK_DECIM,
    GW_BLOCK_INTERP
} gw_block_t;

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
                     gr::io_signature::sptr out_sig);
};

} /* namespace gr */

#endif /* INCLUDED_RUNTIME_BLOCK_GATEWAY_H */
