########################################################################
# Python Bindings
########################################################################

pybind11_add_module(vocoder_python
    alaw_decode_bs_python.cc
    alaw_encode_sb_python.cc
    cvsd_decode_bs_python.cc
    cvsd_encode_sb_python.cc
    g721_decode_bs_python.cc
    g721_encode_sb_python.cc
    g723_24_decode_bs_python.cc
    g723_24_encode_sb_python.cc
    g723_40_decode_bs_python.cc
    g723_40_encode_sb_python.cc
    ulaw_decode_bs_python.cc
    ulaw_encode_sb_python.cc)

    message(status ${LIBCODEC2_FOUND})
    message(status ${LIBCODEC2_HAS_FREEDV_API})
    message(status ${LIBGSM_FOUND})

if(LIBCODEC2_FOUND)
    target_sources(vocoder_python PRIVATE
    codec2_python.cc
    codec2_decode_ps_python.cc
    codec2_encode_sp_python.cc
    )
endif(LIBCODEC2_FOUND)

if(LIBCODEC2_HAS_FREEDV_API)
    target_sources(vocoder_python PRIVATE
    freedv_rx_ss_python.cc
    freedv_tx_ss_python.cc
    )
endif(LIBCODEC2_HAS_FREEDV_API)

if(LIBGSM_FOUND)
    target_sources(vocoder_python PRIVATE
    gsm_fr_decode_ps_python.cc
    gsm_fr_encode_sp_python.cc
    )
endif(LIBGSM_FOUND)

target_sources(vocoder_python PRIVATE
    python_bindings.cc)

target_include_directories(vocoder_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(vocoder_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime gnuradio-vocoder)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(vocoder_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS vocoder_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/vocoder COMPONENT pythonapi)
