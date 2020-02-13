
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
#include <boost/shared_ptr.hpp>
PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

// #include "generated/agc_python.hpp"
// #include "generated/agc2_python.hpp"
#include "generated/agc2_cc_python.hpp"
#include "generated/agc2_ff_python.hpp"
#include "generated/agc3_cc_python.hpp"
#include "generated/agc_cc_python.hpp"
#include "generated/agc_ff_python.hpp"
#include "generated/cpfsk_bc_python.hpp"
#include "generated/cpm_python.hpp"
#include "generated/ctcss_squelch_ff_python.hpp"
#include "generated/dpll_bb_python.hpp"
#include "generated/fastnoise_source_python.hpp"
#include "generated/feedforward_agc_cc_python.hpp"
#include "generated/fmdet_cf_python.hpp"
#include "generated/frequency_modulator_fc_python.hpp"
#include "generated/noise_source_python.hpp"
#include "generated/noise_type_python.hpp"
#include "generated/phase_modulator_fc_python.hpp"
#include "generated/pll_carriertracking_cc_python.hpp"
#include "generated/pll_freqdet_cf_python.hpp"
#include "generated/pll_refout_cc_python.hpp"
#include "generated/probe_avg_mag_sqrd_c_python.hpp"
#include "generated/probe_avg_mag_sqrd_cf_python.hpp"
#include "generated/probe_avg_mag_sqrd_f_python.hpp"
#include "generated/pwr_squelch_cc_python.hpp"
#include "generated/pwr_squelch_ff_python.hpp"
#include "generated/quadrature_demod_cf_python.hpp"
#include "generated/rail_ff_python.hpp"
#include "generated/random_uniform_source_python.hpp"
#include "generated/sig_source_python.hpp"
#include "generated/sig_source_waveform_python.hpp"
#include "generated/simple_squelch_cc_python.hpp"
#include "generated/squelch_base_cc_python.hpp"
#include "generated/squelch_base_ff_python.hpp"

// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void* init_numpy()
{
    import_array();
    return NULL;
}

PYBIND11_MODULE(analog_python, m)
{
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

    py::module::import("gnuradio.gr");

    bind_squelch_base_cc(m);
    bind_squelch_base_ff(m);
    
    // bind_agc(m);
    // bind_agc2(m);
    bind_agc2_cc(m);
    bind_agc2_ff(m);
    bind_agc3_cc(m);
    bind_agc_cc(m);
    bind_agc_ff(m);
    bind_cpfsk_bc(m);
    bind_cpm(m);
    bind_ctcss_squelch_ff(m);
    bind_dpll_bb(m);
    bind_fastnoise_source(m);
    bind_feedforward_agc_cc(m);
    bind_fmdet_cf(m);
    bind_frequency_modulator_fc(m);
    bind_noise_source(m);
    bind_noise_type(m);
    bind_phase_modulator_fc(m);
    bind_pll_carriertracking_cc(m);
    bind_pll_freqdet_cf(m);
    bind_pll_refout_cc(m);
    bind_probe_avg_mag_sqrd_c(m);
    bind_probe_avg_mag_sqrd_cf(m);
    bind_probe_avg_mag_sqrd_f(m);
    bind_pwr_squelch_cc(m);
    bind_pwr_squelch_ff(m);
    bind_quadrature_demod_cf(m);
    bind_rail_ff(m);
    bind_random_uniform_source(m);
    bind_sig_source(m);
    bind_sig_source_waveform(m);
    bind_simple_squelch_cc(m);

}

