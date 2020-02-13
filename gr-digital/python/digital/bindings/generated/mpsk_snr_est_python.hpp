

/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#ifndef INCLUDED_GR_DIGITAL_MPSK_SNR_EST_PYTHON_HPP
#define INCLUDED_GR_DIGITAL_MPSK_SNR_EST_PYTHON_HPP

#include <gnuradio/digital/mpsk_snr_est.h>

void bind_mpsk_snr_est(py::module& m)
{
    using mpsk_snr_est    = gr::digital::mpsk_snr_est;
    using mpsk_snr_est_simple    = gr::digital::mpsk_snr_est_simple;
    using mpsk_snr_est_skew    = gr::digital::mpsk_snr_est_skew;
    using mpsk_snr_est_m2m4    = gr::digital::mpsk_snr_est_m2m4;
    using snr_est_m2m4    = gr::digital::snr_est_m2m4;
    using mpsk_snr_est_svr    = gr::digital::mpsk_snr_est_svr;


    py::class_<mpsk_snr_est,
        std::shared_ptr<mpsk_snr_est>>(m, "mpsk_snr_est")

        .def(py::init<double>(),           py::arg("alpha") 
        )
        .def(py::init<gr::digital::mpsk_snr_est const &>(),           py::arg("arg0") 
        )

        .def("alpha",&mpsk_snr_est::alpha)
        .def("set_alpha",&mpsk_snr_est::set_alpha,
            py::arg("alpha") 
        )
        .def("update",&mpsk_snr_est::update,
            py::arg("noutput_items"), 
            py::arg("input") 
        )
        .def("snr",&mpsk_snr_est::snr)
        .def("signal",&mpsk_snr_est::signal)
        .def("noise",&mpsk_snr_est::noise)
        ;


    py::class_<mpsk_snr_est_simple,gr::digital::mpsk_snr_est,
        std::shared_ptr<mpsk_snr_est_simple>>(m, "mpsk_snr_est_simple")

        .def(py::init<double>(),           py::arg("alpha") 
        )
        .def(py::init<gr::digital::mpsk_snr_est_simple const &>(),           py::arg("arg0") 
        )

        .def("update",&mpsk_snr_est_simple::update,
            py::arg("noutput_items"), 
            py::arg("input") 
        )
        .def("snr",&mpsk_snr_est_simple::snr)
        ;


    py::class_<mpsk_snr_est_skew,gr::digital::mpsk_snr_est,
        std::shared_ptr<mpsk_snr_est_skew>>(m, "mpsk_snr_est_skew")

        .def(py::init<double>(),           py::arg("alpha") 
        )
        .def(py::init<gr::digital::mpsk_snr_est_skew const &>(),           py::arg("arg0") 
        )

        .def("update",&mpsk_snr_est_skew::update,
            py::arg("noutput_items"), 
            py::arg("input") 
        )
        .def("snr",&mpsk_snr_est_skew::snr)
        ;


    py::class_<mpsk_snr_est_m2m4,gr::digital::mpsk_snr_est,
        std::shared_ptr<mpsk_snr_est_m2m4>>(m, "mpsk_snr_est_m2m4")

        .def(py::init<double>(),           py::arg("alpha") 
        )
        .def(py::init<gr::digital::mpsk_snr_est_m2m4 const &>(),           py::arg("arg0") 
        )

        .def("update",&mpsk_snr_est_m2m4::update,
            py::arg("noutput_items"), 
            py::arg("input") 
        )
        .def("snr",&mpsk_snr_est_m2m4::snr)
        ;


    py::class_<snr_est_m2m4,gr::digital::mpsk_snr_est,
        std::shared_ptr<snr_est_m2m4>>(m, "snr_est_m2m4")

        .def(py::init<double,double,double>(),           py::arg("alpha"), 
           py::arg("ka"), 
           py::arg("kw") 
        )
        .def(py::init<gr::digital::snr_est_m2m4 const &>(),           py::arg("arg0") 
        )

        .def("update",&snr_est_m2m4::update,
            py::arg("noutput_items"), 
            py::arg("input") 
        )
        .def("snr",&snr_est_m2m4::snr)
        ;


    py::class_<mpsk_snr_est_svr,gr::digital::mpsk_snr_est,
        std::shared_ptr<mpsk_snr_est_svr>>(m, "mpsk_snr_est_svr")

        .def(py::init<double>(),           py::arg("alpha") 
        )
        .def(py::init<gr::digital::mpsk_snr_est_svr const &>(),           py::arg("arg0") 
        )

        .def("update",&mpsk_snr_est_svr::update,
            py::arg("noutput_items"), 
            py::arg("input") 
        )
        .def("snr",&mpsk_snr_est_svr::snr)
        ;

    py::enum_<gr::digital::snr_est_type_t>(m,"snr_est_type_t")
        .value("SNR_EST_SIMPLE", gr::digital::SNR_EST_SIMPLE) // 0
        .value("SNR_EST_SKEW", gr::digital::SNR_EST_SKEW) // 1
        .value("SNR_EST_M2M4", gr::digital::SNR_EST_M2M4) // 2
        .value("SNR_EST_SVR", gr::digital::SNR_EST_SVR) // 3
        .export_values()
    ;

} 

#endif /* INCLUDED_GR_DIGITAL_MPSK_SNR_EST_PYTHON_HPP */
