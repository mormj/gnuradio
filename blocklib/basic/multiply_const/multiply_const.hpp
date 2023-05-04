#include <gnuradio4/graph.hpp>
#include <gnuradio4/node_traits.hpp>

#include <vir/simd.h>

namespace fg = fair::graph;

namespace gr {
namespace basic {

template <typename T>
class multiply_const : public fg::node<multiply_const<T>,
                                       fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "in">,
                                       fg::OUT<T, 0, std::numeric_limits<std::size_t>::max(), "out">>
{
    T _k = static_cast<T>(1.0f);

public:
    multiply_const() = delete;

    explicit multiply_const(T k, std::string name = fair::graph::this_source_location())
        : _k(k)
    {
        this->set_name(name);
    }

    template <fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto process_one(const V& a) const noexcept
    {
        return a * _k;
    }
};

} // namespace basic
} // namespace gr