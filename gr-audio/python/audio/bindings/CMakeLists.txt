########################################################################
# Python Bindings
########################################################################

pybind11_add_module(audio_python
    sink_python.cc
    source_python.cc
    python_bindings.cc)

target_include_directories(audio_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(audio_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime gnuradio-audio)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(audio_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS audio_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/audio COMPONENT pythonapi)
