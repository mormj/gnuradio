########################################################################
# Python Bindings
########################################################################

pybind11_add_module(fft_python
    # ctrlport_probe_psd_python.cc
    fft_python.cc
    fft_shift_python.cc
    fft_vcc_python.cc
    fft_vfc_python.cc
    goertzel_python.cc
    goertzel_fc_python.cc
    window_python.cc
    python_bindings.cc)

# if(ENABLE_GR_CTRLPORT)
#     target_sources(fft_python PRIVATE
#         ctrlport_probe_psd_python.cc
#     )
# endif(ENABLE_GR_CTRLPORT)

target_include_directories(fft_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(fft_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime   fftw3f::fftw3f
${GR_VOLK_LIB} gnuradio-fft)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(fft_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS fft_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/fft COMPONENT pythonapi)
