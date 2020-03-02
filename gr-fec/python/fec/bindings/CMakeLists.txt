########################################################################
# Python Bindings
########################################################################

pybind11_add_module(fec_python
    async_decoder_python.cc
    async_encoder_python.cc
    ber_bf_python.cc
    cc_decoder_python.cc
    cc_encoder_python.cc
    ccsds_encoder_python.cc
    conv_bit_corr_bb_python.cc
    decode_ccsds_27_fb_python.cc
    decoder_python.cc
    depuncture_bb_python.cc
    dummy_decoder_python.cc
    dummy_encoder_python.cc
    encode_ccsds_27_bb_python.cc
    encoder_python.cc
    fec_mtrx_python.cc
    generic_decoder_python.cc
    generic_encoder_python.cc
    ldpc_G_matrix_python.cc
    ldpc_H_matrix_python.cc
    ldpc_bit_flip_decoder_python.cc
    ldpc_decoder_python.cc
    ldpc_encoder_python.cc
    ldpc_gen_mtrx_encoder_python.cc
    ldpc_par_mtrx_encoder_python.cc
    polar_common_python.cc
    polar_decoder_common_python.cc
    polar_decoder_sc_python.cc
    polar_decoder_sc_list_python.cc
    polar_decoder_sc_systematic_python.cc
    polar_encoder_python.cc
    polar_encoder_systematic_python.cc
    puncture_bb_python.cc
    puncture_ff_python.cc
    repetition_decoder_python.cc
    repetition_encoder_python.cc
    tagged_decoder_python.cc
    tagged_encoder_python.cc
    tpc_common_python.cc
    tpc_decoder_python.cc
    tpc_encoder_python.cc
    python_bindings.cc)

target_include_directories(fec_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(fec_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime gnuradio-fec)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(fec_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS fec_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/fec COMPONENT pythonapi)
