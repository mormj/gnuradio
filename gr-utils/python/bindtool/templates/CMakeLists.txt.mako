${'########################################################################'}
# Python Bindings
${'########################################################################'}
<%
import os
## file_list = files.sort()
%>
pybind11_add_module(${module_name}_python
## File Includes
% for f in files:  
<%
basename = os.path.splitext(f)[0]
%>\
    ${basename}_python.cc
% endfor
    python_bindings.cc)

target_include_directories(${module_name}_python PUBLIC
${'    ${PYTHON_NUMPY_INCLUDE_DIR}'}
${'    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib'}
${'    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include'}
${'    ${PYBIND11_INCLUDE_DIR}'}
)
target_link_libraries(${module_name}_python PUBLIC ${'${Boost_LIBRARIES} ${PYTHON_LIBRARIES}'} gnuradio-runtime gnuradio-${module_name})

if(CMAKE_COMPILER_IS_GNUCC AND NOT APPLE)
    SET_TARGET_PROPERTIES(${module_name}_python PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

install(TARGETS ${module_name}_python DESTINATION ${'${GR_PYTHON_DIR}'}/gnuradio/${module_name} COMPONENT pythonapi)
