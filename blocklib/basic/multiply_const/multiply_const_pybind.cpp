

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio4/basic/multiply_const.hpp>
#include <gnuradio4/graph.hpp>

using namespace gr::basic;
namespace fg = fair::graph;

template <class T>
void bind_multiply_const_template(py::module& m, const char* classname)
{

    py::class_<gr::basic::multiply_const<T>> multiply_const_class(m, classname);

    multiply_const_class.def(py::init([](fg::graph& flow_graph, T k) {
                       return &(flow_graph.make_node<gr::basic::multiply_const<T>>(k));
                   }),
        py::arg("fg"), 
        py::arg("k"))
    ;
}

void bind_multiply_const(py::module& m)
{

// bind_multiply_const_template<std::complex<float>>(m, "multiply_const_cc");
bind_multiply_const_template<float>(m, "multiply_const_ff");

}