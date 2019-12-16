//
// Copyright 2017-2018 Ettus Research, a National Instruments Company
// Copyright 2019 Ettus Research, a National Instruments Brand
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;

// Allow boost::shared_ptr<T> to be a holder class of an object (PyBind11
// supports std::shared_ptr and std::unique_ptr out of the box)
#include <boost/shared_ptr.hpp>
PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

#include "exports/top_block_python.hpp"
#include "exports/io_signature_python.hpp"
#include "exports/hier_block2_python.hpp"
#include "exports/basic_block_python.hpp"

// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
// #if PY_MAJOR_VERSION >= 3
void* init_numpy()
{
    import_array();
    return NULL;
}
// #else
// void init_numpy()
// {
    // import_array();
// }
// #endif

PYBIND11_MODULE(gr_python, m)
{
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

    // Register types submodule
    // export_pmt(m);
    export_top_block(m);
    export_io_signature(m);
    export_hier_block2(m);
    export_basic_block(m);
}

