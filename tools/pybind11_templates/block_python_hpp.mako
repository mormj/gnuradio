
${license}

#ifndef INCLUDED_${modname.upper()}_${blockname.upper()}_PYTHON_HPP
#define INCLUDED_${modname.upper()}_${blockname.upper()}_PYTHON_HPP

#include <gnuradio/${grblocktype}.h>
#include <gnuradio/${modname}/${blockname}.h>

void bind_${blockname}(py::module& m)
{
    using ${blockname}    = gr::${modname}::${blockname};

    py::class_<${blockname}, gr::${grblocktype}, std::shared_ptr<${blockname}>>(m, "${blockname}")
        .def(py::init(&${blockname}::make))
% for method in methods:
        ## ${method}
        .def("${method['name']}",&${blockname}::${method['name']})
% endfor

        .def("to_basic_block",&${blockname}::to_basic_block)
        ;
} 

#endif /* INCLUDED_${modname.upper()}_${blockname.upper()}_PYTHON_HPP */
