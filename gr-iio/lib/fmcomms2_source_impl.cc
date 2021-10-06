/* -*- c++ -*- */
/*
 * Copyright 2014 Analog Devices Inc.
 * Author: Paul Cercueil <paul.cercueil@analog.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "fmcomms2_source_impl.h"
#include <gnuradio/io_signature.h>
#include <ad9361.h>

#include <string>
#include <thread>
#include <vector>

#define MIN_RATE 520333
#define DECINT_RATIO 8
#define OVERFLOW_CHECK_PERIOD_MS 1000

namespace gr {
namespace iio {

template <typename T>
typename fmcomms2_source<T>::sptr fmcomms2_source<T>::make(const std::string& uri,
                                                           const std::vector<bool>& ch_en,
                                                           unsigned long buffer_size)
{
    return gnuradio::make_block_sptr<fmcomms2_source_impl<T>>(
        device_source_impl::get_context(uri), ch_en, buffer_size);
}

template <typename T>
std::vector<std::string> fmcomms2_source_impl<T>::get_channels_vector(bool ch1_en,
                                                                      bool ch2_en,
                                                                      bool ch3_en,
                                                                      bool ch4_en)
{
    std::vector<std::string> channels;
    if (ch1_en)
        channels.push_back("voltage0");
    if (ch2_en)
        channels.push_back("voltage1");
    if (ch3_en)
        channels.push_back("voltage2");
    if (ch4_en)
        channels.push_back("voltage3");
    return channels;
}

template <typename T>
std::vector<std::string>
fmcomms2_source_impl<T>::get_channels_vector(const std::vector<bool>& ch_en)
{
    std::vector<std::string> channels;
    int idx = 0;
    for (auto en : ch_en) {
        if (en) {
            channels.push_back("voltage" + std::to_string(idx));
        }
        idx++;
    }

    return channels;
}

template <typename T>
fmcomms2_source_impl<T>::fmcomms2_source_impl(iio_context* ctx,
                                              const std::vector<bool>& ch_en,
                                              unsigned long buffer_size)
    : gr::sync_block("fmcomms2_source",
                     gr::io_signature::make(0, 0, 0),
                     gr::io_signature::make(1, -1, sizeof(T))),
      device_source_impl(ctx,
                         true,
                         "cf-ad9361-lpc",
                         get_channels_vector(ch_en),
                         "ad9361-phy",
                         std::vector<std::string>(),
                         buffer_size,
                         0)
{
    overflow_thd = std::thread(&fmcomms2_source_impl<T>::check_overflow, this);

    // Device Buffers are always presented as short from device_sink
    d_device_bufs.resize(get_channels_vector(ch_en).size());
    for (size_t i = 0; i < d_device_bufs.size(); i++) {
        d_device_bufs[i].resize(s_initial_device_buf_size);
    }
}

template <typename T>
fmcomms2_source_impl<T>::~fmcomms2_source_impl()
{
    overflow_thd.join();
}

template <typename T>
void fmcomms2_source_impl<T>::check_overflow(void)
{
    uint32_t status;
    int ret;

    // Wait for stream startup
#ifdef _WIN32
    while (thread_stopped) {
        Sleep(OVERFLOW_CHECK_PERIOD_MS);
    }
    Sleep(OVERFLOW_CHECK_PERIOD_MS);
#else
    while (thread_stopped) {
        usleep(OVERFLOW_CHECK_PERIOD_MS * 1000);
    }
    usleep(OVERFLOW_CHECK_PERIOD_MS * 1000);
#endif

    // Clear status registers
    iio_device_reg_write(dev, 0x80000088, 0x6);

    while (!thread_stopped) {
        ret = iio_device_reg_read(dev, 0x80000088, &status);
        if (ret) {
            throw std::runtime_error("Failed to read overflow status register");
        }
        if (status & 4) {
            printf("O");
            // Clear status registers
            iio_device_reg_write(dev, 0x80000088, 4);
        }
#ifdef _WIN32
        Sleep(OVERFLOW_CHECK_PERIOD_MS);
#else
        usleep(OVERFLOW_CHECK_PERIOD_MS * 1000);
#endif
    }
}


template <>
int fmcomms2_source_impl<std::int16_t>::work(int noutput_items,
                                             gr_vector_const_void_star& input_items,
                                             gr_vector_void_star& output_items)
{
    // Since device_source returns shorts, we can just pass off the work
    return device_source_impl::work(noutput_items, input_items, output_items);
}


template <typename T>
int fmcomms2_source_impl<T>::work(int noutput_items,
                                  gr_vector_const_void_star& input_items,
                                  gr_vector_void_star& output_items)
{
    static gr_vector_void_star tmp_output_items;
    if (output_items.size() > tmp_output_items.size()) {
        tmp_output_items.resize(output_items.size());
    }
    for (size_t i = 0; i < output_items.size(); i++) {
        if (2 * noutput_items > (int)d_device_bufs[i].size()) {
            d_device_bufs[i].resize(2 * noutput_items);
        }
        tmp_output_items[i] = static_cast<void*>(d_device_bufs[i].data());
    }

    int ret = device_source_impl::work(2 * noutput_items, input_items, tmp_output_items);

    // Do the conversion from shorts to gr_complex
    for (size_t i = 0; i < output_items.size(); i++) {
        gr_complex* out = (gr_complex*)output_items[i];
        // TODO: use volk
        for (int n = 0; n < noutput_items; n++) {

            out[n] = gr_complex(float(d_device_bufs[i][2 * n]) / 2048.0,
                                float(d_device_bufs[i][2 * n + 1]) / 2048.0);
        }
    }

    if (ret > 0) {
        return ret / 2;
    } else {
        return ret;
    }
}

template <typename T>
void fmcomms2_source_impl<T>::update_dependent_params()
{
    std::vector<std::string> params;
    // Set rate configuration
    if (d_filter_source.compare("Off") == 0) {
        params.push_back("in_voltage_sampling_frequency=" + std::to_string(d_samplerate));
        params.push_back("in_voltage_rf_bandwidth=" + std::to_string(d_bandwidth));
    } else if (d_filter_source.compare("Auto") == 0) {
        int ret = ad9361_set_bb_rate(phy, d_samplerate);
        if (ret) {
            throw std::runtime_error("Unable to set BB rate");
            params.push_back("in_voltage_rf_bandwidth=" + std::to_string(d_bandwidth));
        }
    } else if (d_filter_source.compare("File") == 0) {
        std::string filt(d_filter_filename);
        if (!load_fir_filter(filt, phy))
            throw std::runtime_error("Unable to load filter file");
    } else if (d_filter_source.compare("Design") == 0) {
        int ret = ad9361_set_bb_rate_custom_filter_manual(
            phy, d_samplerate, d_fpass, d_fstop, d_bandwidth, d_bandwidth);
        if (ret) {
            throw std::runtime_error("Unable to set BB rate");
        }
    } else
        throw std::runtime_error("Unknown filter configuration");

    device_source_impl::set_params(params);
    // Filters can only be disabled after the sample rate has been set
    if (d_filter_source.compare("Off") == 0) {
        int ret = ad9361_set_trx_fir_enable(phy, false);
        if (ret) {
            throw std::runtime_error("Unable to disable filters");
        }
    }
}

template <typename T>
void fmcomms2_source_impl<T>::set_len_tag_key(const std::string& len_tag_key)
{
    device_source_impl::set_len_tag_key(len_tag_key);
}

template <typename T>
void fmcomms2_source_impl<T>::set_frequency(unsigned long long frequency)
{
    std::vector<std::string> params;
    params.push_back("out_altvoltage0_RX_LO_frequency=" + std::to_string(frequency));
    device_source_impl::set_params(params);
}

template <typename T>
void fmcomms2_source_impl<T>::set_samplerate(unsigned long samplerate)
{
    std::vector<std::string> params;
    if (samplerate < MIN_RATE) {
        int ret;
        samplerate = samplerate * DECINT_RATIO;
        ret = device_source_impl::handle_decimation_interpolation(
            samplerate, "voltage0", "sampling_frequency", dev, false, false);
        if (ret < 0)
            samplerate = samplerate / 8;
    } else // Disable decimation filter if on
    {
        device_source_impl::handle_decimation_interpolation(
            samplerate, "voltage0", "sampling_frequency", dev, true, false);
    }

    device_source_impl::set_params(params);
    d_samplerate = samplerate;
    update_dependent_params();
}

template <typename T>
void fmcomms2_source_impl<T>::set_gain_mode(size_t chan, const std::string& mode)
{
    bool is_fmcomms4 = !iio_device_find_channel(phy, "voltage1", false);
    if ((!is_fmcomms4 && chan > 0) || chan > 1) {
        throw std::runtime_error("Channel out of range for this device");
    }
    std::vector<std::string> params;

    params.push_back("in_voltage" + std::to_string(chan) +
                     "_gain_control_mode=" + d_gain_mode[chan]);

    device_source_impl::set_params(params);
    d_gain_mode[chan] = mode;
}

template <typename T>
void fmcomms2_source_impl<T>::set_gain(size_t chan, double gain_value)
{
    bool is_fmcomms4 = !iio_device_find_channel(phy, "voltage1", false);
    if ((!is_fmcomms4 && chan > 0) || chan > 1) {
        throw std::runtime_error("Channel out of range for this device");
    }
    std::vector<std::string> params;

    if (d_gain_mode[chan].compare("manual") == 0) {
        std::string gain_string = std::to_string(gain_value);
        std::string::size_type idx = gain_string.find(',');
        if (idx != std::string::npos) // found , as decimal separator, so change to .
            gain_string.replace(idx, 1, ".");
        params.push_back("in_voltage" + std::to_string(chan) +
                         "_hardwaregain=" + gain_string);
    }
    device_source_impl::set_params(params);
    d_gain_value[chan] = gain_value;
}

template <typename T>
void fmcomms2_source_impl<T>::set_quadrature(bool quadrature)
{
    std::vector<std::string> params;
    params.push_back("in_voltage_quadrature_tracking_en=" + std::to_string(quadrature));
    device_source_impl::set_params(params);
    d_quadrature = quadrature;
}

template <typename T>
void fmcomms2_source_impl<T>::set_rfdc(bool rfdc)
{
    std::vector<std::string> params;
    params.push_back("in_voltage_rf_dc_offset_tracking_en=" + std::to_string(rfdc));
    device_source_impl::set_params(params);
    d_rfdc = rfdc;
}

template <typename T>
void fmcomms2_source_impl<T>::set_bbdc(bool bbdc)
{
    std::vector<std::string> params;
    params.push_back("in_voltage_bb_dc_offset_tracking_en=" + std::to_string(bbdc));
    device_source_impl::set_params(params);
    d_bbdc = bbdc;
}

template <typename T>
void fmcomms2_source_impl<T>::set_filter_params(const std::string& filter_source,
                                                const std::string& filter_filename,
                                                float fpass,
                                                float fstop)
{
    d_filter_source = filter_source;
    d_filter_filename = filter_filename;
    d_fpass = fpass;
    d_fstop = fstop;

    update_dependent_params();
}

template class fmcomms2_source<std::int16_t>;
template class fmcomms2_source<gr_complex>;

} /* namespace iio */
} /* namespace gr */
