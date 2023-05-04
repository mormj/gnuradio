#include <gnuradio4/graph.hpp>
#include <gnuradio4/node_traits.hpp>

#include <vir/simd.h>

namespace fg = fair::graph;
using namespace fair::literals;

namespace gr {
namespace basic {


template <typename T>
class vector_source : public fg::node<vector_source<T>,
                                       fg::OUT<T, 0, std::numeric_limits<std::size_t>::max(), "out">>
{
    std::vector<T> _data;
    size_t _n_samples_max;
    size_t _n_samples_produced = 0;

public:
    vector_source() = delete;

    explicit vector_source(const std::vector<T>& data, std::string name = fair::graph::this_source_location())
        : _data(data), _n_samples_max(_data.size()) 
    {
        this->set_name(name);
    }

    [[nodiscard]] constexpr T
    process_one() const noexcept {
        auto ret = _data[_n_samples_produced];
        return ret;
    }

    fair::graph::work_return_t
    work() {
        const std::size_t n_to_publish = _n_samples_max - _n_samples_produced;
        if (n_to_publish > 0) {
            auto &port   = output_port<"out">(this);
            auto &writer = port.streamWriter();

            std::size_t n_write = std::clamp(n_to_publish, 0UL, std::min(writer.available(), port.max_buffer_size()));
            if (n_write == 0_UZ) {
                return fair::graph::work_return_t::INSUFFICIENT_INPUT_ITEMS;
            }

            writer.publish( //
                    [this](std::span<T> output) {
                        for (auto &val : output) {
                            val = process_one();
                            _n_samples_produced++;
                        }
                    },
                    n_write);

            return fair::graph::work_return_t::OK;
        } else {
            return fair::graph::work_return_t::DONE;
        }
    }

};

} // namespace basic
} // namespace gr