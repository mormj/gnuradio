#include <gnuradio/types.h>

namespace gr {


template <>
inline data_type_t get_data_type<gr_complexd>()
{
    return data_type_t::CF64;
}
template <>
inline data_type_t get_data_type<gr_complex>()
{
    return data_type_t::CF32;
}
template <>
inline data_type_t get_data_type<double>()
{
    return data_type_t::RF64;
}
template <>
inline data_type_t get_data_type<float>()
{
    return data_type_t::RF32;
}
template <>
inline data_type_t get_data_type<uint64_t>()
{
    return data_type_t::RU64;
}
template <>
inline data_type_t get_data_type<uint32_t>()
{
    return data_type_t::RU32;
}
template <>
inline data_type_t get_data_type<uint16_t>()
{
    return data_type_t::RU16;
}
template <>
inline data_type_t get_data_type<uint8_t>()
{
    return data_type_t::RU8;
}
template <>
inline data_type_t get_data_type<int64_t>()
{
    return data_type_t::RI64;
}
template <>
inline data_type_t get_data_type<int32_t>()
{
    return data_type_t::RI32;
}
template <>
inline data_type_t get_data_type<int16_t>()
{
    return data_type_t::RI16;
}
template <>
inline data_type_t get_data_type<int8_t>()
{
    return data_type_t::RI8;
}
template <>
inline data_type_t get_data_type<bool>()
{
    return data_type_t::BOOL;
}
template <typename T>
inline data_type_t get_data_type()
{
    throw std::runtime_error("Unsupported data type");
}

} // namespace gr