########################################################################
# Python Bindings
########################################################################
configure_file(${CMAKE_SOURCE_DIR}/docs/doxygen/pydoc_macros.h ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

pybind11_add_module(pmt_python
    pmt_python.cc
    pmt_pool_python.cc
    pmt_sugar_python.cc
    python_bindings.cc)
    
# TODO - replace this with a macro in the cmake/Modules directory
if(ENABLE_DOXYGEN)
    add_custom_command( 
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/docstring_status
        COMMAND python3 ${CMAKE_SOURCE_DIR}/docs/doxygen/update_pydoc.py "sub"
        "--json_path" ${CMAKE_BINARY_DIR}/docs/doxygen/gnuradio_docstrings.json
        "--bindings_dir" ${CMAKE_CURRENT_SOURCE_DIR}
        "--output_dir" ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Adding docstrings into PMT pybind headers ..."
        DEPENDS gnuradio_docstrings)
    add_custom_target(pmt_docstrings ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/docstring_status)
else(ENABLE_DOXYGEN)
    add_custom_command( 
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/docstring_status
        COMMAND python3 ${CMAKE_SOURCE_DIR}/docs/doxygen/update_pydoc.py "copy"
        "--bindings_dir" ${CMAKE_CURRENT_SOURCE_DIR}/docstrings
        "--output_dir" ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Copying PMT docstring templates as pybind headers ...")
    add_custom_target(pmt_docstrings ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/docstring_status)
endif(ENABLE_DOXYGEN)

target_include_directories(pmt_python PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(pmt_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-pmt)
target_compile_options(pmt_python PRIVATE -Wno-unused-variable) # disable 
add_dependencies(pmt_python pmt_docstrings)

install(TARGETS pmt_python DESTINATION ${GR_PYTHON_DIR}/pmt COMPONENT pythonapi)
