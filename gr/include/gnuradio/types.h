#pragma once

#include <gnuradio/gr_complex.h>
#include <pmtf/base.hpp>
#include <cstddef> // size_t
#include <cstdint>
#include <typeindex>
#include <vector>

using gr_vector_int = std::vector<int>;
using gr_vector_uint = std::vector<unsigned int>;
using gr_vector_float = std::vector<float>;
using gr_vector_double = std::vector<double>;
using gr_vector_void_star = std::vector<void*>;
using gr_vector_const_void_star = std::vector<const void*>;


namespace gr {
    enum data_type_t { 
        CF64 = 0,
        CF32,
        RF64,
        RF32,
        RU64,
        RU32,
        RU16,
        RU8,
        RI64,
        RI32,
        RI16,
        RI8,
        SIZE,
        STRING,
        BOOL
    };

    
}

using pmt_sptr = std::shared_ptr<pmtf::pmt>;
