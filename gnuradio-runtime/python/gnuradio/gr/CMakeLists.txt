# Copyright 2012 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

########################################################################
include(GrPython)

GR_PYTHON_INSTALL(FILES
    __init__.py
    exceptions.py
    tag_utils.py
    packet_utils.py
    gateway.py
    hier_block2.py
    top_block.py
    pubsub.py
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/gr
)

########################################################################
# Python Bindings
########################################################################

pybind11_add_module(gr_python ${CMAKE_CURRENT_SOURCE_DIR}/python_bindings.cpp)
target_include_directories(gr_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${PYBIND11_INCLUDE_DIR} 
)
target_link_libraries(gr_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime)

message(STATUS "!!@#$%@#$%^$%&^#%^&#$!@#$!@#$!$^@^%$")
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
get_target_property(OUT gr_python INCLUDE_DIRECTORIES)
message(STATUS ${OUT})

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DESTINATION ${GR_PYTHON_DIR}/gnuradio/ COMPONENT pythonapi)

########################################################################
# Handle the unit tests
########################################################################
if(ENABLE_TESTING)
  set(GR_TEST_TARGET_DEPS "")
  set(GR_TEST_LIBRARY_DIRS "")
  set(GR_TEST_PYTHON_DIRS
    ${CMAKE_BINARY_DIR}/gruel/src/python
    ${CMAKE_BINARY_DIR}/gnuradio-runtime/python
    )
  include(GrTest)
  file(GLOB py_qa_test_files "qa_*.py")
  foreach(py_qa_test_file ${py_qa_test_files})
    get_filename_component(py_qa_test_name ${py_qa_test_file} NAME_WE)
    GR_ADD_TEST(${py_qa_test_name} ${QA_PYTHON_EXECUTABLE} -B ${py_qa_test_file})
  endforeach(py_qa_test_file)
endif(ENABLE_TESTING)
