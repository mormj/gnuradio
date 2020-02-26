########################################################################
# Python Bindings
########################################################################

pybind11_add_module(trellis_python
    # base_python.cc
    constellation_metrics_cf_python.cc
    # core_algorithms_python.cc
    encoder_python.cc
    fsm_python.cc
    interleaver_python.cc
    metrics_python.cc
    pccc_decoder_blk_python.cc
    pccc_decoder_combined_blk_python.cc
    pccc_encoder_python.cc
    permutation_python.cc
    quicksort_index_python.cc
    sccc_decoder_blk_python.cc
    sccc_decoder_combined_blk_python.cc
    sccc_encoder_python.cc
    siso_combined_f_python.cc
    siso_f_python.cc
    siso_type_python.cc
    viterbi_python.cc
    viterbi_combined_python.cc
    python_bindings.cc)

target_include_directories(trellis_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(trellis_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime gnuradio-digital gnuradio-trellis)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(trellis_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS trellis_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/trellis COMPONENT pythonapi)
