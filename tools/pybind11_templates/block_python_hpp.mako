<%
    make_arguments = header_info['make']['arguments']
    blockname = header_info['class']
    modname = header_info['module_name']
    grblocktype = header_info['block_type']
    method_functions=header_info['method_functions']
%>

${license}

#ifndef INCLUDED_${modname.upper()}_${blockname.upper()}_PYTHON_HPP
#define INCLUDED_${modname.upper()}_${blockname.upper()}_PYTHON_HPP

#include <gnuradio/${grblocktype}.h>
#include <gnuradio/${modname}/${blockname}.h>

void bind_${blockname}(py::module& m)
{
    using ${blockname}    = gr::${modname}::${blockname};

    py::class_<${blockname}, gr::${grblocktype}, std::shared_ptr<${blockname}>>(m, "${blockname}")
% if len(make_arguments) == 0:
        .def(py::init(&${blockname}::make)
        )
% else:
        .def(py::init(&${blockname}::make),
% for arg in make_arguments:
            py::arg("${arg['name']}")${" = " + arg['default'] if arg['default'] else ''}${'' if loop.index == len(make_arguments)-1 else ',' } 
% endfor
        )
% endif

% for fcn in method_functions:
<%
fcn_args = fcn['arguments']
%>
% if len(fcn_args) == 0:
        .def("${fcn['name']}",&${blockname}::${fcn['name']})
%else:
        .def("${fcn['name']}",&${blockname}::${fcn['name']},
% for arg in fcn_args:
            py::arg("${arg['name']}")${" = " + arg['default'] if arg['default'] else ''}, 
% endfor
        )
% endif
% endfor
        .def("to_basic_block",[](std::shared_ptr<${blockname}> p){
            return p->to_basic_block();
        })
        ;
} 

#endif /* INCLUDED_${modname.upper()}_${blockname.upper()}_PYTHON_HPP */
