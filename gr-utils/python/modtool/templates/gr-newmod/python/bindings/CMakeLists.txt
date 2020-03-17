# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

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

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(howto_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS howto_python DESTINATION ${GR_PYTHON_DIR}/howto COMPONENT pythonapi)
