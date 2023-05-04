#include <gnuradio4/graph.hpp>
#include <gnuradio4/node_traits.hpp>

#include <vir/simd.h>

namespace fg = fair::graph;
using namespace fair::literals;

namespace gr {
namespace basic {


template <typename T>
class vector_sink
    : public fg::node<vector_sink<T>,
                      fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "in">,
                      fg::OUT<T, 0, std::numeric_limits<std::size_t>::max(), "out">>
{
    std::vector<T> _data;
    size_t _n_samples_max;
    size_t _n_samples_produced = 0;

public:
    // vector_sink() = delete;

    explicit vector_sink(std::string name = fair::graph::this_source_location())
    {
        this->set_name(name);
    }

    fair::graph::work_return_t work()
    {
        auto& port = input_port<"in">(this);
        auto& reader = port.streamReader();

        const auto n_readable = std::min(reader.available(), port.max_buffer_size());

        if (n_readable == 0) {
            return fair::graph::work_return_t::DONE;
        }

        const auto input = reader.get();
        for (std::size_t i = 0; i < n_readable; i++) {
            _data.push_back(input[i]);
        }

        if (!reader.consume(n_readable)) {
            return fair::graph::work_return_t::ERROR;
        }
        return fair::graph::work_return_t::OK;
    }

    std::vector<T>& data() { return _data; }
};

} // namespace basic
} // namespace gr