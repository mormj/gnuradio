

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_OFDM_EQUALIZER_BASE_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_OFDM_EQUALIZER_BASE_PYTHON_HPP

#include <gnuradio/digital/ofdm_equalizer_base.h>

void bind_ofdm_equalizer_base(py::module& m)
{
    using ofdm_equalizer_base    = gr::digital::ofdm_equalizer_base;
    using ofdm_equalizer_1d_pilots    = gr::digital::ofdm_equalizer_1d_pilots;


    py::class_<ofdm_equalizer_base,std::enable_shared_from_this<gr::digital::ofdm_equalizer_base>,
        std::shared_ptr<ofdm_equalizer_base>>(m, "ofdm_equalizer_base")

        // .def(py::init<int>(),           py::arg("fft_len") 
        // )
        // .def(py::init<gr::digital::ofdm_equalizer_base const &>(),           py::arg("arg0") 
        // )

        .def("reset",&ofdm_equalizer_base::reset)
        .def("equalize",&ofdm_equalizer_base::equalize,
            py::arg("frame"), 
            py::arg("n_sym"), 
            py::arg("initial_taps") = std::vector<gr_complex>(), 
            py::arg("tags") = std::vector<gr::tag_t>() 
        )
        .def("get_channel_state",&ofdm_equalizer_base::get_channel_state,
            py::arg("taps") 
        )
        .def("fft_len",&ofdm_equalizer_base::fft_len)
        .def("base",&ofdm_equalizer_base::base)
        ;


    py::class_<ofdm_equalizer_1d_pilots,gr::digital::ofdm_equalizer_base,
        std::shared_ptr<ofdm_equalizer_1d_pilots>>(m, "ofdm_equalizer_1d_pilots")

        // .def(py::init<int,std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const &,std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const &,std::vector<std::vector<std::complex<float>, std::allocator<std::complex<float> > >, std::allocator<std::vector<std::complex<float>, std::allocator<std::complex<float> > > > > const &,int,bool>(),           py::arg("fft_len"), 
        //    py::arg("occupied_carriers"), 
        //    py::arg("pilot_carriers"), 
        //    py::arg("pilot_symbols"), 
        //    py::arg("symbols_skipped"), 
        //    py::arg("input_is_shifted") 
        // )
        // .def(py::init<gr::digital::ofdm_equalizer_1d_pilots const &>(),           py::arg("arg0") 
        // )

        .def("reset",&ofdm_equalizer_1d_pilots::reset)
        .def("get_channel_state",&ofdm_equalizer_1d_pilots::get_channel_state,
            py::arg("taps") 
        )
        ;


} 

#endif /* INCLUDED_GR_DIGITAL_OFDM_EQUALIZER_BASE_PYTHON_HPP */
