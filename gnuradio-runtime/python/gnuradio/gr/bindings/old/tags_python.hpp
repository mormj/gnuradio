//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_GR_TAGS_PYTHON_HPP
#define INCLUDED_GR_TAGS_PYTHON_HPP

// #pragma once

#include <gnuradio/tags.h>

void bind_tags(py::module& m)
{
    using tag_t      = gr::tag_t;

    py::class_<tag_t, std::shared_ptr<tag_t>>(m, "tag_t")
        .def(py::init<>())
        .def(py::init<tag_t>())
        .def_readwrite("offset", &tag_t::offset)
        .def_readwrite("key", &tag_t::key)
        .def_readwrite("value", &tag_t::value)
        .def_readwrite("srcid", &tag_t::srcid)
        .def_readwrite("marked_deleted", &tag_t::marked_deleted)
        // .def(py::self = py::self)
        // .def(py::self == py::self)
        .def("offset_compare", &tag_t::offset_compare)
        ;
} 

#endif /* INCLUDED_GR_TAGS_PYTHON_HPP */
