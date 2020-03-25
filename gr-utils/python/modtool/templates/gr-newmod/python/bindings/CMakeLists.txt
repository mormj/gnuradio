# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Docstrings for Python Bindings
########################################################################

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pydoc_macros.h ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

add_custom_target(howto_docstrings COMMAND
python3 ${PROJECT_SOURCE_DIR}/docs/doxygen/update_pydoc.py
  ${CMAKE_BINARY_DIR}/docs/doxygen/xml
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/howto_docstrings.json
  COMMENT "Adding docstrings into pybind headers ..."
  DEPENDS doxygen-target)


########################################################################
# Python Bindings
########################################################################

pybind11_add_module(howto_python
    python_bindings.cc)

target_include_directories(howto_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${PYBIND11_INCLUDE_DIR}
)
target_link_libraries(howto_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime gnuradio-howto)
add_dependencies(pybindex_python howto_docstrings)

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(howto_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS howto_python DESTINATION ${GR_PYTHON_DIR}/howto COMPONENT pythonapi)
