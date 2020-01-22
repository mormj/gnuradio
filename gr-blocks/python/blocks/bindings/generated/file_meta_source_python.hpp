

/* Copyright 2020 Free Software Foundation, Inc.
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


/* This file is automatically generated using the gen_nonblock_bindings.py tool */

#ifndef INCLUDED_GR_BLOCKS_FILE_META_SOURCE_PYTHON_HPP
#define INCLUDED_GR_BLOCKS_FILE_META_SOURCE_PYTHON_HPP

#include <gnuradio/sync_block.h>
#include <gnuradio/blocks/file_meta_source.h>

void bind_file_meta_source(py::module& m)
{
    using file_meta_source    = gr::blocks::file_meta_source;


    py::class_<file_meta_source,gr::sync_block,
        std::shared_ptr<file_meta_source>>(m, "file_meta_source")

        .def(py::init(&file_meta_source::make),
           py::arg("filename"), 
           py::arg("repeat") = false, 
           py::arg("detached_header") = false, 
           py::arg("hdr_filename") = "" 
        )
        

        .def("open",&file_meta_source::open,
            py::arg("filename"), 
            py::arg("hdr_filename") = "" 
        )
        .def("close",&file_meta_source::close)
        .def("do_update",&file_meta_source::do_update)
        .def("to_basic_block",[](std::shared_ptr<file_meta_source> p){
            return p->to_basic_block();
        })
        ;


} 

#endif /* INCLUDED_GR_BLOCKS_FILE_META_SOURCE_PYTHON_HPP */
