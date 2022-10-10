#include <gnuradio/port_data_type.h>
#include <complex>

namespace gr {

template <class T>
const T& port_data_type_base::get() const
{
    return dynamic_cast<const port_data_type<T>&>(*this).get();
}

template class port_data_type<float>;
template class port_data_type<std::complex<float>>;
template class port_data_type<double>;
template class port_data_type<std::complex<double>>;
template class port_data_type<uint64_t>;
template class port_data_type<uint32_t>;
template class port_data_type<uint16_t>;
template class port_data_type<uint8_t>;
template class port_data_type<int64_t>;
template class port_data_type<int32_t>;
template class port_data_type<int16_t>;
template class port_data_type<int8_t>;
template class port_data_type<bool>;

} // namespace gr
