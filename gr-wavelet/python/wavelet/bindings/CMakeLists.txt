########################################################################
# Python Bindings
########################################################################

pybind11_add_module(wavelet_python
    squash_ff_python.cc
    wavelet_ff_python.cc
    wvps_ff_python.cc
    python_bindings.cc)

# set_target_properties(wavelet_python PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_include_directories(wavelet_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR} 
)
target_link_libraries(wavelet_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime gnuradio-blocks gsl::gsl gnuradio-wavelet)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(wavelet_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

# install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DESTINATION ${GR_PYTHON_DIR}/gnuradio/ COMPONENT pythonapi)
install(TARGETS wavelet_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/wavelet COMPONENT pythonapi)
