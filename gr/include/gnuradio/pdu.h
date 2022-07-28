/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/types.h>
#include <pmtf/map.hpp>
#include <pmtf/vector.hpp>


namespace gr {

/**
 * @brief PDU representation of PMT class
 *
 * Eventually this should go into the PMTF library, but gathering concepts here
 *
 */
class pdu_wrap
{
public:
    pdu_wrap() {}
    // pdu(data_type_t data_type, size_t size, size_t numels) : {

    // }
    // template <typename T>
    // pdu(data_type_t data_type, const std::vector<T>& vec)
    //     : _data((const uint8_t*)vec.data(), (const uint8_t*)(vec.data() + vec.size()))
    // {
    // }
    // template <typename T>
    // pdu(data_type_t data_type, const pmtf::vector<T>& vec)
    //     : _data((const uint8_t*)vec.data(), (const uint8_t*)(vec.data() + vec.size()))
    // {
    // }
    // template <typename T>
    // pdu(data_type_t data_type, std::initializer_list<T> il)
    //     : _data((uint8_t*)il.begin(), (uint8_t*)il.begin() + il.size() * sizeof(T))
    // {
    // }
    pdu_wrap(data_type_t data_type, void* d, size_t size) : _data_type(data_type)
    {
        _data = make_pmt_vector(data_type, d, size);
    }

    // From a Pmt Buffer
    template <class U, typename = pmtf::IsPmt<U>>
    pdu_wrap(const U& other)
    {
        // do better checking here
        auto pmtmap = pmtf::map(other);
        _meta = pmtf::map(pmtmap["meta"]);
        _data = pmtf::vector_wrap(pmtmap["data"]);
    }

    size_t size_bytes() { return _data.bytes(); }
    size_t size() { return _data.bytes() / _data.bytes_per_element(); }
    template <typename T>
    T* data()
    {
        return pmtf::vector<T>(_data).data();
    }
    pmtf::pmt& operator[](const std::string& key) { return _meta[key]; }

    template <typename T>
    T& at(size_t n)
    {
        return data<T>()[n];
    }

    pmtf::pmt get_pmt() const
    {
        auto pmt_map = pmtf::map{ { "meta", _meta }, { "data", _data } };
        return pmt_map;
    }

private:
    data_type_t _data_type;
    pmtf::map _meta;
    pmtf::vector_wrap _data;

    template <typename T>
    static pmtf::pmt _make_pmt_vector(void* data, size_t num_elements)
    {
        return pmtf::vector<T>(static_cast<T*>(data),
                               static_cast<T*>(data) + num_elements);
    }


public:
    /**
     * @brief Convenience method for creating pmt vector from enum
     *
     * @param data_type
     * @param data
     * @param size
     * @return pmtf::pmt
     */
    static pmtf::pmt
    make_pmt_vector(data_type_t data_type, void* data, size_t num_elements)
    {
        switch (data_type) {
        case data_type_t::CF64:
            // return pmtf::vector<std::complex<double>>(static_cast<gr_complexd*>(data),
            // static_cast<gr_complexd*>(data) + num_elements);
            return _make_pmt_vector<gr_complexd>(data, num_elements);
        case data_type_t::CF32:
            return _make_pmt_vector<gr_complex>(data, num_elements);
        case data_type_t::RF64:
            return _make_pmt_vector<double>(data, num_elements);
        case data_type_t::RF32:
            return _make_pmt_vector<float>(data, num_elements);
        case data_type_t::RU64:
            return _make_pmt_vector<uint64_t>(data, num_elements);
        case data_type_t::RU32:
            return _make_pmt_vector<uint32_t>(data, num_elements);
        case data_type_t::RU16:
            return _make_pmt_vector<uint16_t>(data, num_elements);
        case data_type_t::RU8:
            return _make_pmt_vector<uint8_t>(data, num_elements);
        case data_type_t::RI64:
            return _make_pmt_vector<int64_t>(data, num_elements);
        case data_type_t::RI32:
            return _make_pmt_vector<int32_t>(data, num_elements);
        case data_type_t::RI16:
            return _make_pmt_vector<int16_t>(data, num_elements);
        case data_type_t::RI8:
            return _make_pmt_vector<int8_t>(data, num_elements);
        // case data_type_t::BOOL:
        //     return _make_pmt_vector<bool>(data, num_elements);
        default:
            throw std::runtime_error("Invalid PMT vector type requested");
        }
    }

    const void* raw()
    {
        switch (_data_type) {
        case data_type_t::CF64:
            return static_cast<void*>(pmtf::vector<gr_complexd>(_data.get_pmt_buffer()).data());
        case data_type_t::CF32:
            return static_cast<void*>(pmtf::vector<gr_complex>(_data.get_pmt_buffer()).data());
        case data_type_t::RF64:
            return static_cast<void*>(pmtf::vector<double>(_data.get_pmt_buffer()).data());
        case data_type_t::RF32:
            return static_cast<void*>(pmtf::vector<float>(_data.get_pmt_buffer()).data());
        case data_type_t::RU64:
            return static_cast<void*>(pmtf::vector<uint64_t>(_data.get_pmt_buffer()).data());
        case data_type_t::RU32:
            return static_cast<void*>(pmtf::vector<uint32_t>(_data.get_pmt_buffer()).data());
        case data_type_t::RU16:
            return static_cast<void*>(pmtf::vector<uint16_t>(_data.get_pmt_buffer()).data());
        case data_type_t::RU8:
            return static_cast<void*>(pmtf::vector<uint8_t>(_data.get_pmt_buffer()).data());
        case data_type_t::RI64:
            return static_cast<void*>(pmtf::vector<int64_t>(_data.get_pmt_buffer()).data());
        case data_type_t::RI32:
            return static_cast<void*>(pmtf::vector<int32_t>(_data.get_pmt_buffer()).data());
        case data_type_t::RI16:
            return static_cast<void*>(pmtf::vector<int16_t>(_data.get_pmt_buffer()).data());
        case data_type_t::RI8:
            return static_cast<void*>(pmtf::vector<int8_t>(_data.get_pmt_buffer()).data());
        // case data_type_t::BOOL:
        //     return static_cast<void*>(pmtf::vector<bool>(_data.get_pmt_buffer()).data());
        default:
            throw std::runtime_error("Invalid PMT vector type requested");
        }
    }
};

} // namespace gr
