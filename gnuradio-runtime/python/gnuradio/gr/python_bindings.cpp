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
// supports boost::shared_ptr and std::unique_ptr out of the box)
// #include <boost/shared_ptr.hpp>
// PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

#include "bindings/tags_python.hpp"
#include "bindings/top_block_python.hpp"
#include "bindings/io_signature_python.hpp"
#include "bindings/hier_block2_python.hpp"
#include "bindings/basic_block_python.hpp"
#include "bindings/block_python.hpp"
#include "bindings/sync_block_python.hpp"
#include "bindings/sync_decimator_python.hpp"
#include "bindings/sync_interpolator_python.hpp"

#include "bindings/high_res_timer_python.hpp"

#include "bindings/block_detail_python.hpp"
// #include "bindings/block_gateway_python.hpp"
#include "bindings/buffer_python.hpp"
#include "bindings/constants_python.hpp"
#include "bindings/feval_python.hpp"
#include "bindings/random_python.hpp"
// #include "bindings/runtime_types_python.hpp"
#include "bindings/logger_python.hpp"
#include "bindings/message_python.hpp"

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
    bind_tags(m);
    bind_basic_block(m);
    bind_block(m);
    bind_sync_block(m);
    bind_sync_decimator(m);
    bind_sync_interpolator(m);
    bind_io_signature(m);
    bind_hier_block2(m);
    bind_top_block(m);    

    bind_high_res_timer(m);
    bind_block_detail(m);
    // bind_block_gateway(m);
    bind_buffer(m);
    bind_constants(m);
    bind_feval(m);
    bind_random(m);
    // bind_runtime_types(m); // currently empty
    bind_logger(m);
    bind_message(m);
    

    // TODO: Move into gr_types.hpp
    // %constant int sizeof_char 	= sizeof(char);
    m.attr("sizeof_char") = sizeof(char);
    // %constant int sizeof_short	= sizeof(short);
    m.attr("sizeof_short") = sizeof(short);
    // %constant int sizeof_int	= sizeof(int);
    m.attr("sizeof_int") = sizeof(int);
    // %constant int sizeof_float	= sizeof(float);
    m.attr("sizeof_float") = sizeof(float);
    // %constant int sizeof_double	= sizeof(double);
    m.attr("sizeof_double") = sizeof(double);
    // %constant int sizeof_gr_complex	= sizeof(gr_complex);
    m.attr("sizeof_gr_complex") = sizeof(gr_complex);
}

