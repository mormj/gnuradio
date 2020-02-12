

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_FILTER_FFT_FILTER_PYTHON_HPP
#define INCLUDED_GR_FILTER_FFT_FILTER_PYTHON_HPP

#include <gnuradio/filter/fft_filter.h>

void bind_fft_filter(py::module& m)
{
    using fft_filter_fff    = gr::filter::kernel::fft_filter_fff;
    using fft_filter_ccc    = gr::filter::kernel::fft_filter_ccc;
    using fft_filter_ccf    = gr::filter::kernel::fft_filter_ccf;


    py::class_<fft_filter_fff,
        std::shared_ptr<fft_filter_fff>>(m, "fft_filter_fff")

        .def(py::init<int,std::vector<float, std::allocator<float> > const &,int>(),           py::arg("decimation"), 
           py::arg("taps"), 
           py::arg("nthreads") = 1 
        )

        .def("set_taps",&fft_filter_fff::set_taps,
            py::arg("taps") 
        )
        .def("set_nthreads",&fft_filter_fff::set_nthreads,
            py::arg("n") 
        )
        .def("taps",&fft_filter_fff::taps)
        .def("ntaps",&fft_filter_fff::ntaps)
        .def("nthreads",&fft_filter_fff::nthreads)
        .def("filter",&fft_filter_fff::filter,
            py::arg("nitems"), 
            py::arg("input"), 
            py::arg("output") 
        )
        ;


    py::class_<fft_filter_ccc,
        std::shared_ptr<fft_filter_ccc>>(m, "fft_filter_ccc")

        .def(py::init<int,std::vector<std::complex<float>, std::allocator<std::complex<float> > > const &,int>(),           py::arg("decimation"), 
           py::arg("taps"), 
           py::arg("nthreads") = 1 
        )

        .def("set_taps",&fft_filter_ccc::set_taps,
            py::arg("taps") 
        )
        .def("set_nthreads",&fft_filter_ccc::set_nthreads,
            py::arg("n") 
        )
        .def("taps",&fft_filter_ccc::taps)
        .def("ntaps",&fft_filter_ccc::ntaps)
        .def("nthreads",&fft_filter_ccc::nthreads)
        .def("filter",&fft_filter_ccc::filter,
            py::arg("nitems"), 
            py::arg("input"), 
            py::arg("output") 
        )
        ;


    py::class_<fft_filter_ccf,
        std::shared_ptr<fft_filter_ccf>>(m, "fft_filter_ccf")

        .def(py::init<int,std::vector<float, std::allocator<float> > const &,int>(),           py::arg("decimation"), 
           py::arg("taps"), 
           py::arg("nthreads") = 1 
        )

        .def("set_taps",&fft_filter_ccf::set_taps,
            py::arg("taps") 
        )
        .def("set_nthreads",&fft_filter_ccf::set_nthreads,
            py::arg("n") 
        )
        .def("taps",&fft_filter_ccf::taps)
        .def("ntaps",&fft_filter_ccf::ntaps)
        .def("filtersize",&fft_filter_ccf::filtersize)
        .def("nthreads",&fft_filter_ccf::nthreads)
        .def("filter",&fft_filter_ccf::filter,
            py::arg("nitems"), 
            py::arg("input"), 
            py::arg("output") 
        )
        ;


} 

#endif /* INCLUDED_GR_FILTER_FFT_FILTER_PYTHON_HPP */
