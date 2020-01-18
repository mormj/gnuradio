<%
    namespace = header_info['namespace']
    basename = header_info['base_name']
    modname = header_info['module_name']
    classes=header_info['classes']
    free_functions=header_info['free_functions']
%>

${license}

#ifndef INCLUDED_${modname.upper()}_${basename.upper()}_PYTHON_HPP
#define INCLUDED_${modname.upper()}_${basename.upper()}_PYTHON_HPP

#include <${modname}/${basename}.h>

void bind_${basename}(py::module& m)
{
    using ${basename}    = ${"::".join(namespace)}::${basename};
% for cls in classes:
<%
member_functions = cls['member_functions']
%>
    py::class_<${cls['name']}, std::shared_ptr<${cls['name']}>>(m, "${cls['name']}")
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

% for fcn in free_functions:
<%
fcn_args = fcn['arguments']
%>\
% if len(fcn_args) == 0:
        m.def("${fcn['name']}",&${basename}::${fcn['name']});
%else:
        m.def("${fcn['name']}",&${basename}::${fcn['name']},
% for arg in fcn_args:
            py::arg("${arg['name']}")${" = " + arg['default'] if arg['default'] else ''}${'' if loop.index == len(fcn['arguments'])-1 else ',' } 
% endfor
        );
% endif
% endfor

} 

#endif /* INCLUDED_${modname.upper()}_${basename.upper()}_PYTHON_HPP */
