<%
    make_arguments = header_info['make']['arguments']
    blockname = header_info['class']
    modname = header_info['module_name']
    grblocktype = header_info['block_type']
    methods=header_info['methods']
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
        .def(py::init(&${blockname}::make)
% for arg in make_arguments:
            py::arg("${arg['name']}")${" = " + arg['default'] if arg['default'] else ''}${'' if loop.index == len(make_arguments)-1 else ',' } 
% endfor
        )
% endif

        

% for method in methods:
        ## ${method}
        .def("${method['name']}",&${blockname}::${method['name']})
% endfor

        .def("to_basic_block",[](std::shared_ptr<${blockname}> p){
            return p->to_basic_block();
        })
        ;
} 

#endif /* INCLUDED_${modname.upper()}_${blockname.upper()}_PYTHON_HPP */
