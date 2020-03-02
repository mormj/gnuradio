########################################################################
# Python Bindings
########################################################################

pybind11_add_module(zeromq_python
    pub_msg_sink_python.cc
    pub_sink_python.cc
    pull_msg_source_python.cc
    pull_source_python.cc
    push_msg_sink_python.cc
    push_sink_python.cc
    rep_msg_sink_python.cc
    rep_sink_python.cc
    req_msg_source_python.cc
    req_source_python.cc
    sub_msg_source_python.cc
    sub_source_python.cc
    python_bindings.cc)

target_include_directories(zeromq_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(zeromq_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime 
Boost::boost
Boost::thread
ZeroMQ::ZeroMQ 
gnuradio-zeromq)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(zeromq_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS zeromq_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/zeromq COMPONENT pythonapi)
