########################################################################
# Python Bindings
########################################################################

pybind11_add_module(pmt_python
    pmt_python.cc
    pmt_pool_python.cc
    pmt_sugar_python.cc
    python_bindings.cc)

target_include_directories(pmt_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(pmt_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-pmt)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(pmt_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS pmt_python DESTINATION ${GR_PYTHON_DIR}/pmt COMPONENT pythonapi)
