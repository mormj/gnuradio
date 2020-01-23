
/* Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
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

#include "generated/abs_blk_python.hpp"
#include "generated/add_blk_python.hpp"
#include "generated/add_const_bb_python.hpp"
#include "generated/add_const_cc_python.hpp"
#include "generated/add_const_ff_python.hpp"
#include "generated/add_const_ii_python.hpp"
#include "generated/add_const_ss_python.hpp"
#include "generated/add_const_v_python.hpp"
// #include "generated/and_blk_python.hpp"
// #include "generated/and_const_python.hpp"
// #include "generated/annotator_1to1_python.hpp"
// #include "generated/annotator_alltoall_python.hpp"
// #include "generated/annotator_raw_python.hpp"
// // #include "generated/api_python.hpp"
// #include "generated/argmax_python.hpp"
// #include "generated/bin_statistics_f_python.hpp"
#include "generated/burst_tagger_python.hpp"
// #include "generated/char_to_float_python.hpp"
// #include "generated/char_to_short_python.hpp"
// #include "generated/check_lfsr_32k_s_python.hpp"
#include "generated/complex_to_arg_python.hpp"
#include "generated/complex_to_float_python.hpp"
#include "generated/complex_to_imag_python.hpp"
#include "generated/complex_to_interleaved_char_python.hpp"
#include "generated/complex_to_interleaved_short_python.hpp"
#include "generated/complex_to_mag_python.hpp"
#include "generated/complex_to_mag_squared_python.hpp"
#include "generated/complex_to_magphase_python.hpp"
#include "generated/complex_to_real_python.hpp"
// #include "generated/conjugate_cc_python.hpp"
// // #include "generated/control_loop_python.hpp"
#include "generated/copy_python.hpp"
#include "generated/count_bits_python.hpp"
// #include "generated/ctrlport_probe2_b_python.hpp"
// #include "generated/ctrlport_probe2_c_python.hpp"
// #include "generated/ctrlport_probe2_f_python.hpp"
// #include "generated/ctrlport_probe2_i_python.hpp"
// #include "generated/ctrlport_probe2_s_python.hpp"
// #include "generated/ctrlport_probe_c_python.hpp"
#include "generated/deinterleave_python.hpp"
#include "generated/delay_python.hpp"
#include "generated/divide_python.hpp"
// #include "generated/endian_swap_python.hpp"
// #include "generated/exponentiate_const_cci_python.hpp"
// #include "generated/file_descriptor_sink_python.hpp"
// #include "generated/file_descriptor_source_python.hpp"
// #include "generated/file_meta_sink_python.hpp"
// #include "generated/file_meta_source_python.hpp"
// #include "generated/file_sink_python.hpp"
// // #include "generated/file_sink_base_python.hpp"
#include "generated/file_source_python.hpp"
// #include "generated/float_to_char_python.hpp"
// #include "generated/float_to_complex_python.hpp"
// #include "generated/float_to_int_python.hpp"
// #include "generated/float_to_short_python.hpp"
// #include "generated/float_to_uchar_python.hpp"
#include "generated/head_python.hpp"
// #include "generated/int_to_float_python.hpp"
// #include "generated/integrate_python.hpp"
// #include "generated/interleave_python.hpp"
// #include "generated/interleaved_char_to_complex_python.hpp"
// #include "generated/interleaved_short_to_complex_python.hpp"
// #include "generated/keep_m_in_n_python.hpp"
// #include "generated/keep_one_in_n_python.hpp"
// // #include "generated/lfsr_15_1_0_python.hpp"
// // #include "generated/lfsr_32k_python.hpp"
// #include "generated/lfsr_32k_source_s_python.hpp"
// // #include "generated/log2_const_python.hpp"  -- revisit numerical template args
// #include "generated/magphase_to_complex_python.hpp"
// #include "generated/max_blk_python.hpp"
// #include "generated/message_debug_python.hpp"
// #include "generated/message_strobe_python.hpp"
// #include "generated/message_strobe_random_python.hpp"
// #include "generated/min_blk_python.hpp"
// #include "generated/moving_average_python.hpp"
#include "generated/multiply_python.hpp"
// #include "generated/multiply_by_tag_value_cc_python.hpp"
// #include "generated/multiply_conjugate_cc_python.hpp"
#include "generated/multiply_const_python.hpp"
#include "generated/multiply_const_v_python.hpp"
// #include "generated/multiply_matrix_python.hpp"
// #include "generated/mute_python.hpp"
// #include "generated/nlog10_ff_python.hpp"
// #include "generated/nop_python.hpp"
// #include "generated/not_blk_python.hpp"
#include "generated/null_sink_python.hpp"
#include "generated/null_source_python.hpp"
// #include "generated/or_blk_python.hpp"
// // #include "generated/pack_k_bits_python.hpp"
// #include "generated/pack_k_bits_bb_python.hpp"
// #include "generated/packed_to_unpacked_python.hpp"
// #include "generated/patterned_interleaver_python.hpp"
// #include "generated/pdu_python.hpp"
// #include "generated/pdu_filter_python.hpp"
// #include "generated/pdu_remove_python.hpp"
// #include "generated/pdu_set_python.hpp"
// #include "generated/pdu_to_tagged_stream_python.hpp"
// #include "generated/peak_detector_python.hpp"
// #include "generated/peak_detector2_fb_python.hpp"
// #include "generated/plateau_detector_fb_python.hpp"
// #include "generated/probe_rate_python.hpp"
// #include "generated/probe_signal_python.hpp"
// #include "generated/probe_signal_v_python.hpp"
// #include "generated/random_pdu_python.hpp"
// #include "generated/regenerate_bb_python.hpp"
// #include "generated/repack_bits_bb_python.hpp"
// #include "generated/repeat_python.hpp"
// #include "generated/rms_cf_python.hpp"
// #include "generated/rms_ff_python.hpp"
// // #include "generated/rotator_python.hpp"
// #include "generated/rotator_cc_python.hpp"
// #include "generated/sample_and_hold_python.hpp"
// #include "generated/selector_python.hpp"
// #include "generated/short_to_char_python.hpp"
// #include "generated/short_to_float_python.hpp"
#include "generated/skiphead_python.hpp"
// #include "generated/socket_pdu_python.hpp"
// #include "generated/stream_mux_python.hpp"
#include "generated/stream_to_streams_python.hpp"
#include "generated/stream_to_tagged_stream_python.hpp"
#include "generated/stream_to_vector_python.hpp"
#include "generated/streams_to_stream_python.hpp"
#include "generated/streams_to_vector_python.hpp"
// #include "generated/stretch_ff_python.hpp"
#include "generated/sub_python.hpp"
#include "generated/tag_debug_python.hpp"
#include "generated/tag_gate_python.hpp"
#include "generated/tag_share_python.hpp"
#include "generated/tagged_file_sink_python.hpp"
#include "generated/tagged_stream_align_python.hpp"
#include "generated/tagged_stream_multiply_length_python.hpp"
#include "generated/tagged_stream_mux_python.hpp"
#include "generated/tagged_stream_to_pdu_python.hpp"
#include "generated/tags_strobe_python.hpp"
#include "generated/tcp_server_sink_python.hpp"
#include "generated/test_tag_variable_rate_ff_python.hpp"
#include "generated/threshold_ff_python.hpp"
#include "generated/throttle_python.hpp"
#include "generated/transcendental_python.hpp"
#include "generated/tsb_vector_sink_python.hpp"
#include "generated/tuntap_pdu_python.hpp"
#include "generated/uchar_to_float_python.hpp"
#include "generated/udp_sink_python.hpp"
#include "generated/udp_source_python.hpp"
#include "generated/unpack_k_bits_python.hpp"
#include "generated/unpack_k_bits_bb_python.hpp"
#include "generated/unpacked_to_packed_python.hpp"
#include "generated/vco_c_python.hpp"
#include "generated/vco_f_python.hpp"
#include "generated/vector_insert_python.hpp"
#include "generated/vector_map_python.hpp"
#include "custom/vector_sink_python.hpp"
#include "custom/vector_source_python.hpp"
#include "generated/vector_to_stream_python.hpp"
#include "generated/vector_to_streams_python.hpp"
// #include "generated/wavfile_python.hpp"
// #include "generated/wavfile_sink_python.hpp"
// #include "generated/wavfile_source_python.hpp"
// #include "generated/xor_blk_python.hpp"

// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void* init_numpy()
{
    import_array();
    return NULL;
}

PYBIND11_MODULE(blocks_python, m)
{
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

    // Register types submodule
    bind_abs_blk(m);
    bind_add_blk(m);
    bind_add_const_bb(m);
    bind_add_const_cc(m);
    bind_add_const_ff(m);
    bind_add_const_ii(m);
    bind_add_const_ss(m);
    bind_add_const_v(m);
    // bind_and_blk(m);
    // bind_and_const(m);
    // bind_annotator_1to1(m);
    // bind_annotator_alltoall(m);
    // bind_annotator_raw(m);
    // // bind_api(m);
    // bind_argmax(m);
    // bind_bin_statistics_f(m);
    bind_burst_tagger(m);
    // bind_char_to_float(m);
    // bind_char_to_short(m);
    // bind_check_lfsr_32k_s(m);
    bind_complex_to_arg(m);
    bind_complex_to_float(m);
    bind_complex_to_imag(m);
    bind_complex_to_interleaved_char(m);
    bind_complex_to_interleaved_short(m);
    bind_complex_to_mag(m);
    bind_complex_to_mag_squared(m);
    bind_complex_to_magphase(m);
    bind_complex_to_real(m);
    // bind_conjugate_cc(m);
    // // bind_control_loop(m);
    bind_copy(m);
    bind_count_bits(m);
    // bind_ctrlport_probe2_b(m);
    // bind_ctrlport_probe2_c(m);
    // bind_ctrlport_probe2_f(m);
    // bind_ctrlport_probe2_i(m);
    // bind_ctrlport_probe2_s(m);
    // bind_ctrlport_probe_c(m);
    bind_deinterleave(m);
    bind_delay(m);
    bind_divide(m);
    // bind_endian_swap(m);
    // bind_exponentiate_const_cci(m);
    // bind_file_descriptor_sink(m);
    // bind_file_descriptor_source(m);
    // bind_file_meta_sink(m);
    // bind_file_meta_source(m);
    // bind_file_sink(m);
    // // bind_file_sink_base(m);
    bind_file_source(m);
    // bind_float_to_char(m);
    // bind_float_to_complex(m);
    // bind_float_to_int(m);
    // bind_float_to_short(m);
    // bind_float_to_uchar(m);
    bind_head(m);
    // bind_int_to_float(m);
    // bind_integrate(m);
    // bind_interleave(m);
    // bind_interleaved_char_to_complex(m);
    // bind_interleaved_short_to_complex(m);
    // bind_keep_m_in_n(m);
    // bind_keep_one_in_n(m);
    // // bind_lfsr_15_1_0(m);
    // // bind_lfsr_32k(m);
    // bind_lfsr_32k_source_s(m);
    // // bind_log2_const(m);
    // bind_magphase_to_complex(m);
    // bind_max_blk(m);
    // bind_message_debug(m);
    // bind_message_strobe(m);
    // bind_message_strobe_random(m);
    // bind_min_blk(m);
    // bind_moving_average(m);
    bind_multiply(m);
    // bind_multiply_by_tag_value_cc(m);
    // bind_multiply_conjugate_cc(m);
    bind_multiply_const(m);
    bind_multiply_const_v(m);
    // bind_multiply_matrix(m);
    // bind_mute(m);
    // bind_nlog10_ff(m);
    // bind_nop(m);
    // bind_not_blk(m);
    bind_null_sink(m);
    bind_null_source(m);
    // bind_or_blk(m);
    // // bind_pack_k_bits(m);
    // bind_pack_k_bits_bb(m);
    // bind_packed_to_unpacked(m);
    // bind_patterned_interleaver(m);
    // bind_pdu(m);
    // bind_pdu_filter(m);
    // bind_pdu_remove(m);
    // bind_pdu_set(m);
    // bind_pdu_to_tagged_stream(m);
    // bind_peak_detector(m);
    // bind_peak_detector2_fb(m);
    // bind_plateau_detector_fb(m);
    // bind_probe_rate(m);
    // bind_probe_signal(m);
    // bind_probe_signal_v(m);
    // bind_random_pdu(m);
    // bind_regenerate_bb(m);
    // bind_repack_bits_bb(m);
    // bind_repeat(m);
    // bind_rms_cf(m);
    // bind_rms_ff(m);
    // // bind_rotator(m);
    // bind_rotator_cc(m);
    // bind_sample_and_hold(m);
    // bind_selector(m);
    // bind_short_to_char(m);
    // bind_short_to_float(m);
    bind_skiphead(m);
    // bind_socket_pdu(m);
    // bind_stream_mux(m);
    bind_stream_to_streams(m);
    bind_stream_to_tagged_stream(m);
    bind_stream_to_vector(m);
    bind_streams_to_stream(m);
    bind_streams_to_vector(m);
    // bind_stretch_ff(m);
    bind_sub(m);
    bind_tag_debug(m);
    bind_tag_gate(m);
    bind_tag_share(m);
    bind_tagged_file_sink(m);
    bind_tagged_stream_align(m);
    bind_tagged_stream_multiply_length(m);
    bind_tagged_stream_mux(m);
    bind_tagged_stream_to_pdu(m);
    bind_tags_strobe(m);
    bind_tcp_server_sink(m);
    bind_test_tag_variable_rate_ff(m);
    bind_threshold_ff(m);
    bind_throttle(m);
    bind_transcendental(m);
    bind_tsb_vector_sink(m);
    bind_tuntap_pdu(m);
    bind_uchar_to_float(m);
    bind_udp_sink(m);
    bind_udp_source(m);
    bind_unpack_k_bits(m);
    bind_unpack_k_bits_bb(m);
    bind_unpacked_to_packed(m);
    bind_vco_c(m);
    bind_vco_f(m);
    bind_vector_insert(m);
    bind_vector_map(m);
    bind_vector_sink(m);
    bind_vector_source(m);
    bind_vector_to_stream(m);
    bind_vector_to_streams(m);
    // bind_wavfile(m);
    // bind_wavfile_sink(m);
    // bind_wavfile_source(m);
    // bind_xor_blk(m);
}

