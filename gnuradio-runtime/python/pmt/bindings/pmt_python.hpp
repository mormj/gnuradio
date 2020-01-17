//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#ifndef INCLUDED_PMT_PYTHON_HPP
#define INCLUDED_PMT_PYTHON_HPP

// #pragma once

#include <pmt/pmt.h>

void bind_pmt(py::module& m)
{
    using pmt_base      = pmt::pmt_base;
    using pmt_t         = pmt::pmt_t;
    py::class_<pmt_base, std::shared_ptr<pmt_base>>(m, "pmt_t")
        .def(py::init())
        .def("__repr__", &pmt::write_string)
        // virtual bool is_bool() const { return false; }
        .def("is_bool",&pmt::is_bool)
        // virtual bool is_symbol() const { return false; }
        .def("is_symbol",&pmt::is_symbol)
        // virtual bool is_number() const { return false; }
        .def("is_number",&pmt::is_number)
        // virtual bool is_integer() const { return false; }
        .def("is_integer",&pmt::is_integer)
        // virtual bool is_uint64() const { return false; }
        .def("is_uint64",&pmt::is_uint64)
        // virtual bool is_real() const { return false; }
        .def("is_real",&pmt::is_real)
        // virtual bool is_complex() const { return false; }
        .def("is_complex",&pmt::is_complex)
        // virtual bool is_null() const { return false; }
        .def("is_null",&pmt::is_null)
        // virtual bool is_pair() const { return false; }
        .def("is_pair",&pmt::is_pair)
        // virtual bool is_tuple() const { return false; }
        .def("is_tuple",&pmt::is_tuple)
        // virtual bool is_vector() const { return false; }
        .def("is_vector",&pmt::is_vector)
        // virtual bool is_dict() const { return false; }
        .def("is_dict",&pmt::is_dict)
        // virtual bool is_any() const { return false; }
        .def("is_any",&pmt::is_any)

        // virtual bool is_uniform_vector() const { return false; }
        .def("is_uniform_vector",&pmt::is_uniform_vector)
        // virtual bool is_u8vector() const { return false; }
        .def("is_u8vector",&pmt::is_u8vector)
        // virtual bool is_s8vector() const { return false; }
        .def("is_s8vector",&pmt::is_s8vector)
        // virtual bool is_u16vector() const { return false; }
        .def("is_u16vector",&pmt::is_u16vector)
        // virtual bool is_s16vector() const { return false; }
        .def("is_s16vector",&pmt::is_s16vector)
        // virtual bool is_u32vector() const { return false; }
        .def("is_u32vector",&pmt::is_u32vector)
        // virtual bool is_s32vector() const { return false; }
        .def("is_s32vector",&pmt::is_s32vector)
        // virtual bool is_u64vector() const { return false; }
        .def("is_u64vector",&pmt::is_u64vector)
        // virtual bool is_s64vector() const { return false; }
        .def("is_s64vector",&pmt::is_s64vector)
        // virtual bool is_f32vector() const { return false; }
        .def("is_f32vector",&pmt::is_f32vector)
        // virtual bool is_f64vector() const { return false; }
        .def("is_f64vector",&pmt::is_f64vector)
        // virtual bool is_c32vector() const { return false; }
        .def("is_c32vector",&pmt::is_c32vector)
        // virtual bool is_c64vector() const { return false; }
        .def("is_c64vector",&pmt::is_c64vector)
        // .def(py::init<const std::string&, const std::string&>())
        ;


        // PMT_API pmt_t get_PMT_NIL();
        m.def("get_PMT_NIL", &pmt::get_PMT_NIL);
        // PMT_API pmt_t get_PMT_T();
        m.def("get_PMT_T", &pmt::get_PMT_T);
        // PMT_API pmt_t get_PMT_F();
        m.def("get_PMT_F", &pmt::get_PMT_F);
        // PMT_API pmt_t get_PMT_EOF();
        m.def("get_PMT_EOF", &pmt::get_PMT_EOF);

        // #define PMT_NIL get_PMT_NIL()
        m.attr("PMT_NIL") = pmt::get_PMT_NIL();
        // #define PMT_T get_PMT_T()
        m.attr("get_PMT_T") = pmt::get_PMT_T();
        // #define PMT_F get_PMT_F()
        m.attr("get_PMT_F") = pmt::get_PMT_F();
        // #define PMT_EOF get_PMT_EOF()
        m.attr("get_PMT_EOF") = pmt::get_PMT_EOF();

    // py::class_<pmt_t>(m, "pmt_t")
        // ;
    // py::class_<pmt::exception>(m,"exception");
    
    m.def("is_bool", &pmt::is_bool);
    m.def("to_bool", &pmt::to_bool);
    m.def("from_bool", &pmt::from_bool);
    m.def("string_to_symbol", &pmt::string_to_symbol);
    // m.def("add", &add, "A function which adds two numbers");
    // m.def("asdf", &pmt::asdf, "A function which adds two numbers");

} 

#endif /* INCLUDED_UHD_USRP_MULTI_USRP_PYTHON_HPP */
