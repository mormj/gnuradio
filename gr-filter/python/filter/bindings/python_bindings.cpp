
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

#include "generated/dc_blocker_cc_python.hpp"
#include "generated/dc_blocker_ff_python.hpp"
#include "generated/fft_filter_python.hpp"
#include "generated/fft_filter_ccc_python.hpp"
#include "generated/fft_filter_ccf_python.hpp"
#include "generated/fft_filter_fff_python.hpp"
#include "generated/filter_delay_fc_python.hpp"
#include "generated/filterbank_python.hpp"
#include "generated/filterbank_vcvcf_python.hpp"
#include "generated/fir_filter_python.hpp"
#include "generated/fir_filter_blk_python.hpp"
#include "generated/fir_filter_with_buffer_python.hpp"
#include "generated/firdes_python.hpp"
#include "generated/freq_xlating_fir_filter_python.hpp"
#include "generated/hilbert_fc_python.hpp"
#include "generated/iir_filter_ccc_python.hpp"
#include "generated/iir_filter_ccd_python.hpp"
#include "generated/iir_filter_ccf_python.hpp"
#include "generated/iir_filter_ccz_python.hpp"
#include "generated/iir_filter_ffd_python.hpp"
#include "generated/interp_fir_filter_python.hpp"
#include "generated/mmse_fir_interpolator_cc_python.hpp"
#include "generated/mmse_fir_interpolator_ff_python.hpp"
#include "generated/mmse_interp_differentiator_cc_python.hpp"
#include "generated/mmse_interp_differentiator_ff_python.hpp"
#include "generated/mmse_interpolator_cc_python.hpp"
#include "generated/mmse_interpolator_ff_python.hpp"
#include "generated/mmse_resampler_cc_python.hpp"
#include "generated/mmse_resampler_ff_python.hpp"
#include "generated/pfb_arb_resampler_ccc_python.hpp"
#include "generated/pfb_arb_resampler_ccf_python.hpp"
#include "generated/pfb_arb_resampler_fff_python.hpp"
#include "generated/pfb_channelizer_ccf_python.hpp"
#include "generated/pfb_decimator_ccf_python.hpp"
#include "generated/pfb_interpolator_ccf_python.hpp"
#include "generated/pfb_synthesizer_ccf_python.hpp"
#include "generated/pm_remez_python.hpp"
#include "generated/polyphase_filterbank_python.hpp"
#include "generated/rational_resampler_base_python.hpp"
#include "generated/single_pole_iir_python.hpp"
#include "generated/single_pole_iir_filter_cc_python.hpp"
#include "generated/single_pole_iir_filter_ff_python.hpp"


// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void* init_numpy()
{
    import_array();
    return NULL;
}

PYBIND11_MODULE(filter_python, m)
{
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

    py::module::import("gnuradio.gr");

    bind_dc_blocker_cc(m);
    bind_dc_blocker_ff(m);
    // bind_fft_filter(m);
    bind_fft_filter_ccc(m);
    bind_fft_filter_ccf(m);
    bind_fft_filter_fff(m);
    bind_filter_delay_fc(m);
    bind_filterbank(m);
    bind_filterbank_vcvcf(m);
    // bind_fir_filter(m);
    bind_fir_filter_blk(m);
    bind_fir_filter_with_buffer(m);
    bind_firdes(m);
    bind_freq_xlating_fir_filter(m);
    bind_hilbert_fc(m);
    bind_iir_filter_ccc(m);
    bind_iir_filter_ccd(m);
    bind_iir_filter_ccf(m);
    bind_iir_filter_ccz(m);
    bind_iir_filter_ffd(m);
    bind_interp_fir_filter(m);
    bind_mmse_fir_interpolator_cc(m);
    bind_mmse_fir_interpolator_ff(m);
    bind_mmse_interp_differentiator_cc(m);
    bind_mmse_interp_differentiator_ff(m);
    bind_mmse_interpolator_cc(m);
    bind_mmse_interpolator_ff(m);
    bind_mmse_resampler_cc(m);
    bind_mmse_resampler_ff(m);
    bind_pfb_arb_resampler_ccc(m);
    bind_pfb_arb_resampler_ccf(m);
    bind_pfb_arb_resampler_fff(m);
    bind_pfb_channelizer_ccf(m);
    bind_pfb_decimator_ccf(m);
    bind_pfb_interpolator_ccf(m);
    bind_pfb_synthesizer_ccf(m);
    bind_pm_remez(m);
    bind_polyphase_filterbank(m);
    bind_rational_resampler_base(m);
    bind_single_pole_iir(m);
    bind_single_pole_iir_filter_cc(m);
    bind_single_pole_iir_filter_ff(m);
}

