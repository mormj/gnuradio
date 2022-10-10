#pragma once

namespace gr {

class port_data_type_base
{
    public:
        template <typename U>
        void bulk_convert(const void *buf_in, void *buf_out);
        template <typename U>
        bool is_convertible();
        template <typename U>
        bool conversion_needed();
    private:
        template<class T> const T& get() const;
};

template <typename T>
class port_data_type : port_data_type_base
{
    public:
        template <typename U>
        bool is_convertible();
    private:
        T t;
};

}