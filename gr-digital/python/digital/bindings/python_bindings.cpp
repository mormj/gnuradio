
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

#include "generated/additive_scrambler_bb_python.hpp"
#include "generated/binary_slicer_fb_python.hpp"
#include "generated/burst_shaper_python.hpp"
#include "generated/chunks_to_symbols_python.hpp"
#include "generated/clock_recovery_mm_cc_python.hpp"
#include "generated/clock_recovery_mm_ff_python.hpp"
#include "generated/cma_equalizer_cc_python.hpp"
#include "generated/constellation_python.hpp"
#include "generated/constellation_decoder_cb_python.hpp"
#include "generated/constellation_receiver_cb_python.hpp"
#include "generated/constellation_soft_decoder_cf_python.hpp"
#include "generated/corr_est_cc_python.hpp"
#include "generated/correlate_access_code_bb_python.hpp"
#include "generated/correlate_access_code_bb_ts_python.hpp"
#include "generated/correlate_access_code_ff_ts_python.hpp"
#include "generated/correlate_access_code_tag_bb_python.hpp"
#include "generated/correlate_access_code_tag_ff_python.hpp"
#include "generated/costas_loop_cc_python.hpp"
#include "generated/cpmmod_bc_python.hpp"
#include "generated/crc32_python.hpp"
#include "generated/crc32_async_bb_python.hpp"
#include "generated/crc32_bb_python.hpp"
#include "generated/descrambler_bb_python.hpp"
#include "generated/diff_decoder_bb_python.hpp"
#include "generated/diff_encoder_bb_python.hpp"
#include "generated/diff_phasor_cc_python.hpp"
#include "generated/fll_band_edge_cc_python.hpp"
#include "generated/framer_sink_1_python.hpp"
#include "generated/glfsr_python.hpp"
#include "generated/glfsr_source_b_python.hpp"
#include "generated/glfsr_source_f_python.hpp"
#include "generated/hdlc_deframer_bp_python.hpp"
#include "generated/hdlc_framer_pb_python.hpp"
#include "generated/header_buffer_python.hpp"
#include "generated/header_format_base_python.hpp"
#include "generated/header_format_counter_python.hpp"
#include "generated/header_format_crc_python.hpp"
#include "generated/header_format_default_python.hpp"
#include "generated/header_format_ofdm_python.hpp"
#include "generated/header_payload_demux_python.hpp"
#include "generated/interpolating_resampler_type_python.hpp"
#include "generated/kurtotic_equalizer_cc_python.hpp"
#include "generated/lfsr_python.hpp"
#include "generated/lms_dd_equalizer_cc_python.hpp"
#include "generated/map_bb_python.hpp"
#include "generated/metric_type_python.hpp"
// #include "generated/modulate_vector_python.hpp"
#include "generated/mpsk_snr_est_python.hpp"
#include "generated/mpsk_snr_est_cc_python.hpp"
#include "generated/msk_timing_recovery_cc_python.hpp"
#include "generated/ofdm_carrier_allocator_cvc_python.hpp"
#include "generated/ofdm_chanest_vcvc_python.hpp"
#include "generated/ofdm_cyclic_prefixer_python.hpp"
#include "generated/ofdm_equalizer_base_python.hpp"
#include "generated/ofdm_equalizer_simpledfe_python.hpp"
#include "generated/ofdm_equalizer_static_python.hpp"
#include "generated/ofdm_frame_equalizer_vcvc_python.hpp"
#include "generated/ofdm_serializer_vcc_python.hpp"
#include "generated/ofdm_sync_sc_cfb_python.hpp"
#include "generated/packet_header_default_python.hpp"
#include "generated/packet_header_ofdm_python.hpp"
#include "generated/packet_headergenerator_bb_python.hpp"
#include "generated/packet_headerparser_b_python.hpp"
#include "generated/packet_sink_python.hpp"
#include "generated/pfb_clock_sync_ccf_python.hpp"
#include "generated/pfb_clock_sync_fff_python.hpp"
#include "generated/pn_correlator_cc_python.hpp"
#include "generated/probe_density_b_python.hpp"
#include "generated/probe_mpsk_snr_est_c_python.hpp"
#include "generated/protocol_formatter_async_python.hpp"
#include "generated/protocol_formatter_bb_python.hpp"
#include "generated/protocol_parser_b_python.hpp"
#include "generated/scrambler_bb_python.hpp"
#include "generated/simple_correlator_python.hpp"
#include "generated/simple_framer_python.hpp"
#include "generated/simple_framer_sync_python.hpp"
#include "generated/symbol_sync_cc_python.hpp"
#include "generated/symbol_sync_ff_python.hpp"
#include "generated/timing_error_detector_type_python.hpp"

// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void* init_numpy()
{
    import_array();
    return NULL;
}

PYBIND11_MODULE(digital_python, m)
{
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

    py::module::import("gnuradio.gr");

    // Register types submodule
    bind_additive_scrambler_bb(m);
    bind_binary_slicer_fb(m);
    bind_burst_shaper(m);
    bind_chunks_to_symbols(m);
    bind_clock_recovery_mm_cc(m);
    bind_clock_recovery_mm_ff(m);
    bind_cma_equalizer_cc(m);
    bind_constellation(m);
    bind_constellation_decoder_cb(m);
    bind_constellation_receiver_cb(m);
    bind_constellation_soft_decoder_cf(m);
    bind_corr_est_cc(m);
    bind_correlate_access_code_bb(m);
    bind_correlate_access_code_bb_ts(m);
    bind_correlate_access_code_ff_ts(m);
    bind_correlate_access_code_tag_bb(m);
    bind_correlate_access_code_tag_ff(m);
    bind_costas_loop_cc(m);
    bind_cpmmod_bc(m);
    bind_crc32(m);
    bind_crc32_async_bb(m);
    bind_crc32_bb(m);
    bind_descrambler_bb(m);
    bind_diff_decoder_bb(m);
    bind_diff_encoder_bb(m);
    bind_diff_phasor_cc(m);
    bind_fll_band_edge_cc(m);
    bind_framer_sink_1(m);
    bind_glfsr(m);
    bind_glfsr_source_b(m);
    bind_glfsr_source_f(m);
    bind_hdlc_deframer_bp(m);
    bind_hdlc_framer_pb(m);
    bind_header_buffer(m);
    bind_header_format_base(m);
    bind_header_format_counter(m);
    bind_header_format_crc(m);
    bind_header_format_default(m);
    bind_header_format_ofdm(m);
    bind_header_payload_demux(m);
    bind_interpolating_resampler_type(m);
    bind_kurtotic_equalizer_cc(m);
    bind_lfsr(m);
    bind_lms_dd_equalizer_cc(m);
    bind_map_bb(m);
    bind_metric_type(m);
    // bind_modulate_vector(m);
    bind_mpsk_snr_est(m);
    bind_mpsk_snr_est_cc(m);
    bind_msk_timing_recovery_cc(m);
    bind_ofdm_carrier_allocator_cvc(m);
    bind_ofdm_chanest_vcvc(m);
    bind_ofdm_cyclic_prefixer(m);
    bind_ofdm_equalizer_base(m);
    bind_ofdm_equalizer_simpledfe(m);
    bind_ofdm_equalizer_static(m);
    bind_ofdm_frame_equalizer_vcvc(m);
    bind_ofdm_serializer_vcc(m);
    bind_ofdm_sync_sc_cfb(m);
    bind_packet_header_default(m);
    bind_packet_header_ofdm(m);
    bind_packet_headergenerator_bb(m);
    bind_packet_headerparser_b(m);
    bind_packet_sink(m);
    bind_pfb_clock_sync_ccf(m);
    bind_pfb_clock_sync_fff(m);
    bind_pn_correlator_cc(m);
    bind_probe_density_b(m);
    bind_probe_mpsk_snr_est_c(m);
    bind_protocol_formatter_async(m);
    bind_protocol_formatter_bb(m);
    bind_protocol_parser_b(m);
    bind_scrambler_bb(m);
    bind_simple_correlator(m);
    bind_simple_framer(m);
    bind_simple_framer_sync(m);
    bind_symbol_sync_cc(m);
    bind_symbol_sync_ff(m);
    bind_timing_error_detector_type(m);
}

