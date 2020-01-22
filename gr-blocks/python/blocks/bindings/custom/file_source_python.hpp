

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


#ifndef INCLUDED_BLOCKS_FILE_SOURCE_PYTHON_HPP
#define INCLUDED_BLOCKS_FILE_SOURCE_PYTHON_HPP

#include <gnuradio/sync_block.h>
#include <gnuradio/blocks/file_source.h>

void bind_file_source(py::module& m)
{
    using file_source    = gr::blocks::file_source;

    py::class_<file_source, gr::sync_block, std::shared_ptr<file_source>>(m, "file_source")
        .def(py::init(&file_source::make),
            py::arg("itemsize"), 
            py::arg("filename"), 
            py::arg("repeat") = false, 
            py::arg("offset") = 0, 
            py::arg("len") = 0 
        )


        .def("seek",&file_source::seek,
            py::arg("seek_point"), 
            py::arg("whence")
        )

        .def("open",&file_source::open,
            py::arg("filename"), 
            py::arg("repeat"), 
            py::arg("offset") = 0, 
            py::arg("len") = 0
        )

        .def("close",&file_source::close)

        .def("set_begin_tag",&file_source::set_begin_tag,
            py::arg("val")
        )
        .def("to_basic_block",[](std::shared_ptr<file_source> p){
            return p->to_basic_block();
        })
        ;
} 

#endif /* INCLUDED_BLOCKS_FILE_SOURCE_PYTHON_HPP */
