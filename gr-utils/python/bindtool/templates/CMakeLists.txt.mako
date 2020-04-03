include(GrPybind)

${'########################################################################'}
# Python Bindings
${'########################################################################'}
<%
import os
## file_list = files.sort()
%>
list(APPEND ${module_name}_python_files
## File Includes
% for f in files:  
<%
basename = os.path.splitext(f)[0]
%>\
    ${basename}_python.cc
% endfor
    python_bindings.cc)

GR_PYBIND_MAKE(${module_name} 
   ../../.. 
   "\$\{${module_name}_python_files\}")

install(TARGETS ${module_name}_python DESTINATION ${'${GR_PYTHON_DIR}'}/gnuradio/${module_name} COMPONENT pythonapi)
