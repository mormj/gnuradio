<%
    namespace = header_info['namespace']
    modname = header_info['module_name']
    classes=header_info['classes']
    free_functions=header_info['free_functions']
%>

${license}

#ifndef INCLUDED_${'_'.join(namespace).upper()}_${basename.upper()}_PYTHON_HPP
#define INCLUDED_${'_'.join(namespace).upper()}_${basename.upper()}_PYTHON_HPP

#include <${modname}/${basename}.h>

void bind_${basename}(py::module& m)
{
% if classes:
    using ${basename}    = ${"::".join(namespace)}::${basename};
% endif ##classes
% for cls in classes:
<%
try:
        member_functions = cls['member_functions']
except:
        member_functions = []
try:
        constructors = cls['constructors']
except:
        constructors = []
%>
    py::class_<${cls['name']}, std::shared_ptr<${cls['name']}>>(m, "${cls['name']}")
% for fcn in constructors:
<%
fcn_args = fcn['arguments']
%>\
% if len(fcn_args) == 0:
        .def(py::init<>())
%else:
        .def(py::init<\
% for arg in fcn_args:
${arg['dtype']}${'>(),' if loop.index == len(fcn['arguments'])-1 else ',' }\
% endfor ## args

% for arg in fcn_args:
           py::arg("${arg['name']}")${" = " + arg['default'] if arg['default'] else ''}${'' if loop.index == len(fcn['arguments'])-1 else ',' } 
% endfor
        )
% endif
% endfor ## constructors
% for fcn in member_functions:
<%
fcn_args = fcn['arguments']
%>\
% if len(fcn_args) == 0:
        .def("${fcn['name']}",&${basename}::${fcn['name']})
%else:
        .def("${fcn['name']}",&${basename}::${fcn['name']},
% for arg in fcn_args:
            py::arg("${arg['name']}")${" = " + arg['default'] if arg['default'] else ''}${'' if loop.index == len(fcn['arguments'])-1 else ',' } 
% endfor ## args
        )
% endif
% endfor ## member_functions
        ;
% endfor ## classes

% if free_functions:
% for fcn in free_functions:
<%
fcn_args = fcn['arguments']
%>\
% if len(fcn_args) == 0:
    m.def("${fcn['name']}",&${'::'.join(namespace)}::${fcn['name']});
%else:
    m.def("${fcn['name']}",&${'::'.join(namespace)}::${fcn['name']},
% for arg in fcn_args:
        py::arg("${arg['name']}")${" = " + arg['default'] if arg['default'] else ''}${'' if loop.index == len(fcn['arguments'])-1 else ',' } 
% endfor
    );
% endif
% endfor
% endif ## free_functions
} 

#endif /* INCLUDED_${'_'.join(namespace).upper()}_${basename.upper()}_PYTHON_HPP */
