
/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */



#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;

// Allow boost::shared_ptr<T> to be a holder class of an object (PyBind11
// supports boost::shared_ptr and std::unique_ptr out of the box)
// #include <boost/shared_ptr.hpp>
// PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

#include "generated/fft_python.hpp"
#include "generated/fft_shift_python.hpp"
#include "generated/goertzel_python.hpp"
#include "generated/goertzel_fc_python.hpp"
// #include "generated/ctrlport_probe_psd_python.hpp"
#include "generated/fft_vcc_python.hpp"
#include "generated/fft_vfc_python.hpp"
#include "generated/window_python.hpp"


// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void* init_numpy()
{
    import_array();
    return NULL;
}

PYBIND11_MODULE(fft_python, m)
{
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

     py::module::import("gnuradio.gr");

    // Register types submodule
    bind_fft(m);
    bind_fft_shift(m);
    bind_goertzel(m);
    bind_goertzel_fc(m);
    // bind_ctrlport_probe_psd(m);
    bind_fft_vcc(m);
    bind_fft_vfc(m);
    bind_window(m);
}
