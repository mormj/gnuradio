#include <gnuradio4/graph.hpp>
#include <gnuradio4/basic/multiply_const.hpp>
#include <gnuradio4/basic/vector_source.hpp>
#include <gnuradio4/basic/vector_sink.hpp>

#include <numeric>

namespace fg = fair::graph;

int main() {
    fg::graph flow_graph;

    size_t n_samples = 100e6;
    float k = 7.0;
    
    std::vector<float> d(n_samples);
    std::iota(d.begin(), d.end(), 1);

    auto     &src  = flow_graph.make_node<gr::basic::vector_source<float>>(d);
    auto     &snk = flow_graph.make_node<gr::basic::vector_sink<float>>();
    auto     &mc = flow_graph.make_node<gr::basic::multiply_const<float>>(k);

    auto res = flow_graph.connect<"out">(src).to<"in">(mc);
    res = flow_graph.connect<"out">(mc).to<"in">(snk);

    auto token = flow_graph.init();
    flow_graph.work(token);

    auto d_out = snk.data();

    for (size_t i=0; i<n_samples; i++) {
        if (d_out[i] != k * d[i]) {
            std::cerr << d_out[i] << " != " << d[i] << std::endl;
        }
    }

    return 0;
}