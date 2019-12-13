# Copyright 2011,2013 Free Software Foundation, Inc.
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

include(GrPython)

########################################################################
# Install python files
########################################################################
GR_PYTHON_INSTALL(FILES
    __init__.py
    # pmt_to_python.py
    DESTINATION ${GR_PYTHON_DIR}/pmt
)

########################################################################
# Python Bindings
########################################################################
add_library(tmp SHARED tmp.cpp)
target_link_libraries(tmp ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-runtime )

# Get include dirs
# include_directories(${PYTHON_INCLUDE_DIRS})
# set(PYBIND11_INCLUDE_DIR
#     "${CMAKE_SOURCE_DIR}/deps/pybind11/include"
#     CACHE
#     STRING
#     "Location of PyBind11 includes"
# )
# include_directories(${PYBIND11_INCLUDE_DIR})
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
    "from __future__ import print_function\ntry:\n import numpy\n import os\n inc_path = numpy.get_include()\n if os.path.exists(os.path.join(inc_path, 'numpy', 'arrayobject.h')):\n  print(inc_path, end='')\nexcept:\n pass"
    OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR)

    message(STATUS "^^^%^%^%^^^^%^%^%^^^^^^^^^^^^^^^%^^%^^^^^^^^^^^^^%%^")
message(STATUS ${PYTHON_NUMPY_INCLUDE_DIR})

# message(STATUS "^^^%^%^%^^^^%^%^%^^^^^^^^^^^^^^^%^^%^^^^^^^^^^^^^%%^")
# message(STATUS ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

# Build pyuhd library
# add_library(pmt_python SHARED ${CMAKE_CURRENT_SOURCE_DIR}/python_bindings.cpp)

# target_include_directories(pmt_python PUBLIC
#     ${PYTHON_NUMPY_INCLUDE_DIR}
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include
#     ${PYBIND11_INCLUDE_DIR}
# )

# # target_link_libraries(pmt_python ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-pmt )
# target_link_libraries(pmt_python ${Boost_LIBRARIES} gnuradio-pmt pybind11::module )
# set_target_properties(pmt_python PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
#                                          SUFFIX "${PYTHON_MODULE_EXTENSION}")
# Copy pyuhd library to the staging directory


# pybind11_add_module(pmt_python ${CMAKE_CURRENT_SOURCE_DIR}/python_bindings.cpp)
# target_include_directories(pmt_python PUBLIC
#     ${PYTHON_NUMPY_INCLUDE_DIR}
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include
#     ${PYBIND11_INCLUDE_DIR} 
# )
# target_link_libraries(pmt_python PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} gnuradio-pmt)

message(STATUS "^^^%^%^%^^^^%^%^%^^^^^^^^^^^^^^^%^^%^^^^^^^^^^^^^%%^")
message(STATUS ${PYTHON_LIBRARIES})

add_library(pmt_python SHARED python_bindings.cpp)
target_include_directories(pmt_python PUBLIC
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${PYBIND11_INCLUDE_DIR} 
)
SET(PMT_LINK_LIBRARIES -Wl,--no-as-needed gnuradio-pmt)
# /share/gnuradio/grnext/src/gnuradio/build/gnuradio-runtime/lib/pmt/libgnuradio-pmt.so.3.9.0.0-git
target_link_libraries(pmt_python  ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} pybind11::module ${PMT_LINK_LIBRARIES})
set_target_properties(pmt_python PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

# if(WIN32)
#     set(PMT_PYTHON_LIBRARY_NAME libpmt_python.pyd)
# else()
#     set(PMT_PYTHON_LIBRARY_NAME libpmt_python${CMAKE_SHARED_MODULE_SUFFIX})
# endif(WIN32)

# add_custom_command(TARGET pmt_python
#     POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pmt_python> ${CMAKE_CURRENT_BINARY_DIR}/${PMT_PYTHON_LIBRARY_NAME})

message(STATUS ${CMAKE_INSTALL_PREFIX})

message(STATUS "Utilizing the python install directory: ${PMT_PYTHON_DIR}")
message(STATUS ${CMAKE_CURRENT_BINARY_DIR})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DESTINATION ${GR_PYTHON_DIR} COMPONENT pythonapi)

# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()

get_target_property(OUT pmt_python LINK_LIBRARIES)
message(STATUS ${OUT})

########################################################################
# Setup unit tests
########################################################################
# if(ENABLE_TESTING)
# include(GrTest)
# file(GLOB py_qa_test_files "qa_*.py")
# foreach(py_qa_test_file ${py_qa_test_files})
#     get_filename_component(py_qa_test_name ${py_qa_test_file} NAME_WE)
#     set(GR_TEST_PYTHON_DIRS
#         ${CMAKE_BINARY_DIR}/gnuradio-runtime/python
#         ${CMAKE_BINARY_DIR}/gnuradio-runtime/swig
#         ${CMAKE_BINARY_DIR}/gnuradio-blocks/swig		
#     )
#     set(GR_TEST_TARGET_DEPS gnuradio-runtime)
#     GR_ADD_TEST(${py_qa_test_name} ${QA_PYTHON_EXECUTABLE} -B ${py_qa_test_file})
# endforeach(py_qa_test_file)
# endif(ENABLE_TESTING)
