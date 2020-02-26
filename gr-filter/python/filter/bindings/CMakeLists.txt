########################################################################
# Python Bindings
########################################################################

pybind11_add_module(filter_python
    dc_blocker_cc_python.cc
    dc_blocker_ff_python.cc
    fft_filter_ccc_python.cc
    fft_filter_ccf_python.cc
    fft_filter_fff_python.cc
    filter_delay_fc_python.cc
    filterbank_python.cc
    filterbank_vcvcf_python.cc
    fir_filter_blk_python.cc
    fir_filter_with_buffer_python.cc
    firdes_python.cc
    freq_xlating_fir_filter_python.cc
    hilbert_fc_python.cc
    iir_filter_ccc_python.cc
    iir_filter_ccd_python.cc
    iir_filter_ccf_python.cc
    iir_filter_ccz_python.cc
    iir_filter_ffd_python.cc
    # interp_differentiator_taps_python.cc
    interp_fir_filter_python.cc
    # interpolator_taps_python.cc
    mmse_fir_interpolator_cc_python.cc
    mmse_fir_interpolator_ff_python.cc
    mmse_interp_differentiator_cc_python.cc
    mmse_interp_differentiator_ff_python.cc
    mmse_interpolator_cc_python.cc
    mmse_interpolator_ff_python.cc
    mmse_resampler_cc_python.cc
    mmse_resampler_ff_python.cc
    pfb_arb_resampler_ccc_python.cc
    pfb_arb_resampler_ccf_python.cc
    pfb_arb_resampler_fff_python.cc
    pfb_channelizer_ccf_python.cc
    pfb_decimator_ccf_python.cc
    pfb_interpolator_ccf_python.cc
    pfb_synthesizer_ccf_python.cc
    pm_remez_python.cc
    polyphase_filterbank_python.cc
    rational_resampler_base_python.cc
    single_pole_iir_python.cc
    single_pole_iir_filter_cc_python.cc
    single_pole_iir_filter_ff_python.cc
    python_bindings.cc)

target_include_directories(filter_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(filter_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime gnuradio-filter)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(filter_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS filter_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/filter COMPONENT pythonapi)
