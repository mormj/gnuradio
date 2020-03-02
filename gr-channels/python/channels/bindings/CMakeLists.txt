########################################################################
# Python Bindings
########################################################################

pybind11_add_module(channels_python
    cfo_model_python.cc
    channel_model_python.cc
    channel_model2_python.cc
    dynamic_channel_model_python.cc
    fading_model_python.cc
    selective_fading_model_python.cc
    selective_fading_model2_python.cc
    sro_model_python.cc
    python_bindings.cc)

target_include_directories(channels_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(channels_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime gnuradio-channels)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(channels_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS channels_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/channels COMPONENT pythonapi)
