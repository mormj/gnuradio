

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_CHUNKS_TO_SYMBOLS_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_CHUNKS_TO_SYMBOLS_PYTHON_HPP

#include <gnuradio/digital/chunks_to_symbols.h>

template <class IN_T, class OUT_T>
void bind_chunks_to_symbols_template(py::module& m, const char *classname)
{
    using chunks_to_symbols      = gr::digital::chunks_to_symbols<IN_T,OUT_T>;

    py::class_<chunks_to_symbols, gr::sync_interpolator, std::shared_ptr<chunks_to_symbols>>(m, classname)
        .def(py::init(&gr::digital::chunks_to_symbols<IN_T,OUT_T>::make),
            py::arg("symbol_table"),
            py::arg("D") = 1
        )

        .def("set_symbol_table",&chunks_to_symbols::set_symbol_table,py::arg("symbol_table"))
        .def("symbol_table",&chunks_to_symbols::symbol_table)
        .def("D",&chunks_to_symbols::D)

        .def("to_basic_block",[](std::shared_ptr<chunks_to_symbols> p){
            return p->to_basic_block();
        })
        ;

} 

void bind_chunks_to_symbols(py::module& m)
{
    bind_chunks_to_symbols_template<std::uint8_t, float>(m,"chunks_to_symbols_bf");
    bind_chunks_to_symbols_template<std::uint8_t, gr_complex>(m,"chunks_to_symbols_bc");
    bind_chunks_to_symbols_template<std::int16_t, float>(m,"chunks_to_symbols_sf");
    bind_chunks_to_symbols_template<std::int16_t, gr_complex>(m,"chunks_to_symbols_sc");
    bind_chunks_to_symbols_template<std::int32_t, float>(m,"chunks_to_symbols_if");
    bind_chunks_to_symbols_template<std::int32_t, gr_complex>(m,"chunks_to_symbols_ic");
} 

#endif /* INCLUDED_GR_DIGITAL_CHUNKS_TO_SYMBOLS_PYTHON_HPP */
