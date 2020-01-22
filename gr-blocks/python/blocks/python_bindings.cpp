
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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;

// Allow boost::shared_ptr<T> to be a holder class of an object (PyBind11
// supports boost::shared_ptr and std::unique_ptr out of the box)
#include <boost/shared_ptr.hpp>
PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

#include "bindings/tcp_server_sink_python.hpp"
#include "bindings/file_meta_sink_python.hpp"
#include "bindings/unpack_k_bits_python.hpp"
#include "bindings/log2_const_python.hpp"
#include "bindings/mute_python.hpp"
#include "bindings/int_to_float_python.hpp"
#include "bindings/tag_gate_python.hpp"
#include "bindings/pdu_python.hpp"
#include "bindings/patterned_interleaver_python.hpp"
#include "bindings/complex_to_mag_squared_python.hpp"
#include "bindings/complex_to_interleaved_short_python.hpp"
#include "bindings/complex_to_arg_python.hpp"
#include "bindings/complex_to_imag_python.hpp"
#include "bindings/vector_to_streams_python.hpp"
#include "bindings/wavfile_sink_python.hpp"
#include "bindings/ctrlport_probe2_c_python.hpp"
#include "bindings/copy_python.hpp"
#include "bindings/null_sink_python.hpp"
#include "bindings/magphase_to_complex_python.hpp"
#include "bindings/short_to_float_python.hpp"
#include "bindings/tsb_vector_sink_python.hpp"
#include "bindings/exponentiate_const_cci_python.hpp"
#include "bindings/wavfile_source_python.hpp"
#include "bindings/sub_python.hpp"
#include "bindings/tag_debug_python.hpp"
#include "bindings/add_const_cc_python.hpp"
#include "bindings/pack_k_bits_bb_python.hpp"
#include "bindings/ctrlport_probe2_s_python.hpp"
#include "bindings/peak_detector_python.hpp"
#include "bindings/repeat_python.hpp"
#include "bindings/stream_to_streams_python.hpp"
#include "bindings/endian_swap_python.hpp"
#include "bindings/pdu_set_python.hpp"
#include "bindings/vector_to_stream_python.hpp"
#include "bindings/float_to_uchar_python.hpp"
#include "bindings/xor_blk_python.hpp"
#include "bindings/ctrlport_probe2_f_python.hpp"
#include "bindings/tag_share_python.hpp"
// #include "bindings/lfsr_15_1_0_python.hpp"
#include "bindings/multiply_const_python.hpp"
#include "bindings/stretch_ff_python.hpp"
#include "bindings/annotator_raw_python.hpp"
#include "bindings/probe_rate_python.hpp"
#include "bindings/socket_pdu_python.hpp"
#include "bindings/multiply_python.hpp"
// #include "bindings/lfsr_32k_python.hpp"
#include "bindings/vector_map_python.hpp"
#include "bindings/sample_and_hold_python.hpp"
#include "bindings/complex_to_real_python.hpp"
#include "bindings/tags_strobe_python.hpp"
#include "bindings/vco_c_python.hpp"
#include "bindings/check_lfsr_32k_s_python.hpp"
#include "bindings/min_blk_python.hpp"
#include "bindings/streams_to_vector_python.hpp"
#include "bindings/moving_average_python.hpp"
#include "bindings/probe_signal_python.hpp"
#include "bindings/rotator_cc_python.hpp"
#include "bindings/wavfile_python.hpp"
#include "bindings/multiply_by_tag_value_cc_python.hpp"
#include "bindings/nop_python.hpp"
#include "bindings/pdu_remove_python.hpp"
#include "bindings/or_blk_python.hpp"
#include "bindings/annotator_alltoall_python.hpp"
#include "bindings/complex_to_float_python.hpp"
#include "bindings/repack_bits_bb_python.hpp"
#include "bindings/streams_to_stream_python.hpp"
#include "bindings/multiply_matrix_python.hpp"
#include "bindings/multiply_conjugate_cc_python.hpp"
// #include "bindings/file_sink_base_python.hpp"
#include "bindings/add_const_ss_python.hpp"
// #include "bindings/rotator_python.hpp"
#include "bindings/complex_to_interleaved_char_python.hpp"
#include "bindings/tagged_stream_mux_python.hpp"
#include "bindings/uchar_to_float_python.hpp"
#include "bindings/ctrlport_probe2_i_python.hpp"
#include "bindings/unpacked_to_packed_python.hpp"
#include "bindings/probe_signal_v_python.hpp"
#include "bindings/keep_m_in_n_python.hpp"
#include "bindings/divide_python.hpp"
#include "bindings/float_to_short_python.hpp"
#include "bindings/add_const_ii_python.hpp"
#include "bindings/add_const_v_python.hpp"
#include "bindings/transcendental_python.hpp"
#include "bindings/pdu_to_tagged_stream_python.hpp"
#include "bindings/test_tag_variable_rate_ff_python.hpp"
#include "bindings/udp_source_python.hpp"
#include "bindings/not_blk_python.hpp"
#include "bindings/integrate_python.hpp"
#include "bindings/pdu_filter_python.hpp"
#include "bindings/unpack_k_bits_bb_python.hpp"
#include "bindings/bin_statistics_f_python.hpp"
#include "bindings/vector_sink_python.hpp"
#include "bindings/packed_to_unpacked_python.hpp"
#include "bindings/float_to_int_python.hpp"
#include "bindings/and_const_python.hpp"
#include "bindings/tagged_file_sink_python.hpp"
#include "bindings/multiply_const_v_python.hpp"
#include "bindings/rms_ff_python.hpp"
#include "bindings/file_descriptor_source_python.hpp"
#include "bindings/nlog10_ff_python.hpp"
#include "bindings/random_pdu_python.hpp"
#include "bindings/message_strobe_random_python.hpp"
#include "bindings/add_blk_python.hpp"
#include "bindings/rms_cf_python.hpp"
#include "bindings/vco_f_python.hpp"
#include "bindings/add_const_ff_python.hpp"
#include "bindings/lfsr_32k_source_s_python.hpp"
#include "bindings/udp_sink_python.hpp"
#include "bindings/max_blk_python.hpp"
#include "bindings/delay_python.hpp"
#include "bindings/file_meta_source_python.hpp"
#include "bindings/char_to_float_python.hpp"
// #include "bindings/pack_k_bits_python.hpp"
#include "bindings/keep_one_in_n_python.hpp"
#include "bindings/selector_python.hpp"
#include "bindings/short_to_char_python.hpp"
#include "bindings/stream_to_vector_python.hpp"
#include "bindings/file_sink_python.hpp"
#include "bindings/add_const_bb_python.hpp"
#include "bindings/stream_mux_python.hpp"
#include "bindings/stream_to_tagged_stream_python.hpp"
#include "bindings/tagged_stream_to_pdu_python.hpp"
#include "bindings/interleaved_short_to_complex_python.hpp"
#include "bindings/file_descriptor_sink_python.hpp"
#include "bindings/null_source_python.hpp"
#include "bindings/complex_to_magphase_python.hpp"
#include "bindings/float_to_char_python.hpp"
#include "bindings/tagged_stream_multiply_length_python.hpp"
#include "bindings/tuntap_pdu_python.hpp"
#include "bindings/complex_to_mag_python.hpp"
#include "bindings/regenerate_bb_python.hpp"
#include "bindings/argmax_python.hpp"
#include "bindings/skiphead_python.hpp"
#include "bindings/message_debug_python.hpp"
// #include "bindings/control_loop_python.hpp"
#include "bindings/interleave_python.hpp"
#include "bindings/threshold_ff_python.hpp"
#include "bindings/throttle_python.hpp"
#include "bindings/burst_tagger_python.hpp"
#include "bindings/ctrlport_probe2_b_python.hpp"
#include "bindings/file_source_python.hpp"
#include "bindings/vector_insert_python.hpp"
#include "bindings/vector_source_python.hpp"
#include "bindings/tagged_stream_align_python.hpp"
#include "bindings/deinterleave_python.hpp"
#include "bindings/char_to_short_python.hpp"
#include "bindings/message_strobe_python.hpp"
#include "bindings/head_python.hpp"
#include "bindings/float_to_complex_python.hpp"
#include "bindings/and_blk_python.hpp"
#include "bindings/abs_blk_python.hpp"
// #include "bindings/api_python.hpp"
#include "bindings/conjugate_cc_python.hpp"
#include "bindings/annotator_1to1_python.hpp"
#include "bindings/plateau_detector_fb_python.hpp"
#include "bindings/ctrlport_probe_c_python.hpp"
#include "bindings/count_bits_python.hpp"
#include "bindings/interleaved_char_to_complex_python.hpp"
#include "bindings/peak_detector2_fb_python.hpp"

// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void* init_numpy()
{
    import_array();
    return NULL;
}

PYBIND11_MODULE(gr_python, m)
{
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

    // Register types submodule
    bind_tcp_server_sink(m);
    bind_file_meta_sink(m);
    bind_unpack_k_bits(m);
    bind_log2_const(m);
    bind_mute(m);
    bind_int_to_float(m);
    bind_tag_gate(m);
    bind_pdu(m);
    bind_patterned_interleaver(m);
    bind_complex_to_mag_squared(m);
    bind_complex_to_interleaved_short(m);
    bind_complex_to_arg(m);
    bind_complex_to_imag(m);
    bind_vector_to_streams(m);
    bind_wavfile_sink(m);
    bind_ctrlport_probe2_c(m);
    bind_copy(m);
    bind_null_sink(m);
    bind_magphase_to_complex(m);
    bind_short_to_float(m);
    bind_tsb_vector_sink(m);
    bind_exponentiate_const_cci(m);
    bind_wavfile_source(m);
    bind_sub(m);
    bind_tag_debug(m);
    bind_add_const_cc(m);
    bind_pack_k_bits_bb(m);
    bind_ctrlport_probe2_s(m);
    bind_peak_detector(m);
    bind_repeat(m);
    bind_stream_to_streams(m);
    bind_endian_swap(m);
    bind_pdu_set(m);
    bind_vector_to_stream(m);
    bind_float_to_uchar(m);
    bind_xor_blk(m);
    bind_ctrlport_probe2_f(m);
    bind_tag_share(m);
    bind_lfsr_15_1_0(m);
    bind_multiply_const(m);
    bind_stretch_ff(m);
    bind_annotator_raw(m);
    bind_probe_rate(m);
    bind_socket_pdu(m);
    bind_multiply(m);
    bind_lfsr_32k(m);
    bind_vector_map(m);
    bind_sample_and_hold(m);
    bind_complex_to_real(m);
    bind_tags_strobe(m);
    bind_vco_c(m);
    bind_check_lfsr_32k_s(m);
    bind_min_blk(m);
    bind_streams_to_vector(m);
    bind_moving_average(m);
    bind_probe_signal(m);
    bind_rotator_cc(m);
    bind_wavfile(m);
    bind_multiply_by_tag_value_cc(m);
    bind_nop(m);
    bind_pdu_remove(m);
    bind_or_blk(m);
    bind_annotator_alltoall(m);
    bind_complex_to_float(m);
    bind_repack_bits_bb(m);
    bind_streams_to_stream(m);
    bind_multiply_matrix(m);
    bind_multiply_conjugate_cc(m);
    bind_file_sink_base(m);
    bind_add_const_ss(m);
    bind_rotator(m);
    bind_complex_to_interleaved_char(m);
    bind_tagged_stream_mux(m);
    bind_uchar_to_float(m);
    bind_ctrlport_probe2_i(m);
    bind_unpacked_to_packed(m);
    bind_probe_signal_v(m);
    bind_keep_m_in_n(m);
    bind_divide(m);
    bind_float_to_short(m);
    bind_add_const_ii(m);
    bind_add_const_v(m);
    bind_transcendental(m);
    bind_pdu_to_tagged_stream(m);
    bind_test_tag_variable_rate_ff(m);
    bind_udp_source(m);
    bind_not_blk(m);
    bind_integrate(m);
    bind_pdu_filter(m);
    bind_unpack_k_bits_bb(m);
    bind_bin_statistics_f(m);
    bind_vector_sink(m);
    bind_packed_to_unpacked(m);
    bind_float_to_int(m);
    bind_and_const(m);
    bind_tagged_file_sink(m);
    bind_multiply_const_v(m);
    bind_rms_ff(m);
    bind_file_descriptor_source(m);
    bind_nlog10_ff(m);
    bind_random_pdu(m);
    bind_message_strobe_random(m);
    bind_add_blk(m);
    bind_rms_cf(m);
    bind_vco_f(m);
    bind_add_const_ff(m);
    bind_lfsr_32k_source_s(m);
    bind_udp_sink(m);
    bind_max_blk(m);
    bind_delay(m);
    bind_file_meta_source(m);
    bind_char_to_float(m);
    bind_pack_k_bits(m);
    bind_keep_one_in_n(m);
    bind_selector(m);
    bind_short_to_char(m);
    bind_stream_to_vector(m);
    bind_file_sink(m);
    bind_add_const_bb(m);
    bind_stream_mux(m);
    bind_stream_to_tagged_stream(m);
    bind_tagged_stream_to_pdu(m);
    bind_interleaved_short_to_complex(m);
    bind_file_descriptor_sink(m);
    bind_null_source(m);
    bind_complex_to_magphase(m);
    bind_float_to_char(m);
    bind_tagged_stream_multiply_length(m);
    bind_tuntap_pdu(m);
    bind_complex_to_mag(m);
    bind_regenerate_bb(m);
    bind_argmax(m);
    bind_skiphead(m);
    bind_message_debug(m);
    bind_control_loop(m);
    bind_interleave(m);
    bind_threshold_ff(m);
    bind_throttle(m);
    bind_burst_tagger(m);
    bind_ctrlport_probe2_b(m);
    bind_file_source(m);
    bind_vector_insert(m);
    bind_vector_source(m);
    bind_tagged_stream_align(m);
    bind_deinterleave(m);
    bind_char_to_short(m);
    bind_message_strobe(m);
    bind_head(m);
    bind_float_to_complex(m);
    bind_and_blk(m);
    bind_abs_blk(m);
    bind_api(m);
    bind_conjugate_cc(m);
    bind_annotator_1to1(m);
    bind_plateau_detector_fb(m);
    bind_ctrlport_probe_c(m);
    bind_count_bits(m);
    bind_interleaved_char_to_complex(m);
    bind_peak_detector2_fb(m);
}

