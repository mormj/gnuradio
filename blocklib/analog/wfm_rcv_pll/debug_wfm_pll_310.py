#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: josh
# GNU Radio version: v3.11.0.0git-245-g09930fdc

from gnuradio import analog
import math
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import math as pmath


def snipfcn_snippet_0(self):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.plot(np.real(self.snk.data()))
    plt.plot(np.imag(self.snk.data()))
    plt.figure()
    plt.plot(np.real(self.snk2.data()))
    plt.plot(np.imag(self.snk2.data()))

    plt.show()


def snippets_main_after_stop(tb):
    snipfcn_snippet_0(tb)


class debug_wfm_pll_310(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 768000
        self.rf_decim = rf_decim = 2
        self.demod_rate = demod_rate = (int)(samp_rate/rf_decim)
        self.stereo_carrier_filter_coeffs = stereo_carrier_filter_coeffs = firdes.band_pass( -2.0, demod_rate, 37600, 38400, 400, window.WIN_HAMMING, 6.76)
        self.pilot_carrier_filter_coeffs = pilot_carrier_filter_coeffs = firdes.complex_band_pass( 1.0, demod_rate, 18980, 19020, 1500, window.WIN_HAMMING, 6.76)
        self.audio_decim = audio_decim = (int)(demod_rate/48000)
        self.samp_delay = samp_delay = (len( pilot_carrier_filter_coeffs) - 1) // 2 + (len(stereo_carrier_filter_coeffs) - 1) // 2
        self.deviation = deviation = 75000
        self.deemph_tau = deemph_tau = 75e-6
        self.audio_rate = audio_rate = demod_rate / audio_decim
        self.audio_filter_coeffs = audio_filter_coeffs = firdes.low_pass(1, demod_rate, 15000, 1500, window.WIN_HAMMING, 6.76)

        ##################################################
        # Blocks
        ##################################################
        self.snk2 = blocks.vector_sink_f(1, 1024)
        self.snk = blocks.vector_sink_c(1, 1024)
        self.fir_filter_xxx_0 = filter.fir_filter_fcc(1, pilot_carrier_filter_coeffs)
        self.fir_filter_xxx_0.declare_sample_delay(0)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_head_0_0 = blocks.head(gr.sizeof_float*1, 10000)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, 10000)
        self.blocks_complex_to_imag_0 = blocks.complex_to_imag(1)
        self.analog_sig_source_x_0 = analog.sig_source_c(demod_rate, analog.GR_COS_WAVE, 1000, 1, 0, 0)
        self.analog_quadrature_demod_cf_0 = analog.quadrature_demod_cf((demod_rate / (2 * pmath.pi * deviation)))
        self.analog_pll_refout_cc_0 = analog.pll_refout_cc(0.001, (2 * pmath.pi * 18800 / demod_rate), (2 * pmath.pi * 19200 / demod_rate))


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_pll_refout_cc_0, 0), (self.blocks_head_0, 0))
        self.connect((self.analog_pll_refout_cc_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.analog_pll_refout_cc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.analog_quadrature_demod_cf_0, 0), (self.fir_filter_xxx_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.analog_quadrature_demod_cf_0, 0))
        self.connect((self.blocks_complex_to_imag_0, 0), (self.blocks_head_0_0, 0))
        self.connect((self.blocks_head_0, 0), (self.snk, 0))
        self.connect((self.blocks_head_0_0, 0), (self.snk2, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_complex_to_imag_0, 0))
        self.connect((self.fir_filter_xxx_0, 0), (self.analog_pll_refout_cc_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_demod_rate((int)(self.samp_rate/self.rf_decim))

    def get_rf_decim(self):
        return self.rf_decim

    def set_rf_decim(self, rf_decim):
        self.rf_decim = rf_decim
        self.set_demod_rate((int)(self.samp_rate/self.rf_decim))

    def get_demod_rate(self):
        return self.demod_rate

    def set_demod_rate(self, demod_rate):
        self.demod_rate = demod_rate
        self.set_audio_decim((int)(self.demod_rate/48000))
        self.set_audio_filter_coeffs(firdes.low_pass(1, self.demod_rate, 15000, 1500, window.WIN_HAMMING, 6.76))
        self.set_audio_rate(self.demod_rate / self.audio_decim)
        self.set_pilot_carrier_filter_coeffs(firdes.complex_band_pass( 1.0, self.demod_rate, 18980, 19020, 1500, window.WIN_HAMMING, 6.76))
        self.set_stereo_carrier_filter_coeffs(firdes.band_pass( -2.0, self.demod_rate, 37600, 38400, 400, window.WIN_HAMMING, 6.76))
        self.analog_pll_refout_cc_0.set_max_freq((2 * pmath.pi * 18800 / self.demod_rate))
        self.analog_pll_refout_cc_0.set_min_freq((2 * pmath.pi * 19200 / self.demod_rate))
        self.analog_quadrature_demod_cf_0.set_gain((self.demod_rate / (2 * pmath.pi * self.deviation)))
        self.analog_sig_source_x_0.set_sampling_freq(self.demod_rate)

    def get_stereo_carrier_filter_coeffs(self):
        return self.stereo_carrier_filter_coeffs

    def set_stereo_carrier_filter_coeffs(self, stereo_carrier_filter_coeffs):
        self.stereo_carrier_filter_coeffs = stereo_carrier_filter_coeffs
        self.set_samp_delay((len( self.pilot_carrier_filter_coeffs) - 1) // 2 + (len(self.stereo_carrier_filter_coeffs) - 1) // 2)

    def get_pilot_carrier_filter_coeffs(self):
        return self.pilot_carrier_filter_coeffs

    def set_pilot_carrier_filter_coeffs(self, pilot_carrier_filter_coeffs):
        self.pilot_carrier_filter_coeffs = pilot_carrier_filter_coeffs
        self.set_samp_delay((len( self.pilot_carrier_filter_coeffs) - 1) // 2 + (len(self.stereo_carrier_filter_coeffs) - 1) // 2)
        self.fir_filter_xxx_0.set_taps(self.pilot_carrier_filter_coeffs)

    def get_audio_decim(self):
        return self.audio_decim

    def set_audio_decim(self, audio_decim):
        self.audio_decim = audio_decim
        self.set_audio_rate(self.demod_rate / self.audio_decim)

    def get_samp_delay(self):
        return self.samp_delay

    def set_samp_delay(self, samp_delay):
        self.samp_delay = samp_delay

    def get_deviation(self):
        return self.deviation

    def set_deviation(self, deviation):
        self.deviation = deviation
        self.analog_quadrature_demod_cf_0.set_gain((self.demod_rate / (2 * pmath.pi * self.deviation)))

    def get_deemph_tau(self):
        return self.deemph_tau

    def set_deemph_tau(self, deemph_tau):
        self.deemph_tau = deemph_tau

    def get_audio_rate(self):
        return self.audio_rate

    def set_audio_rate(self, audio_rate):
        self.audio_rate = audio_rate

    def get_audio_filter_coeffs(self):
        return self.audio_filter_coeffs

    def set_audio_filter_coeffs(self, audio_filter_coeffs):
        self.audio_filter_coeffs = audio_filter_coeffs




def main(top_block_cls=debug_wfm_pll_310, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        snippets_main_after_stop(tb)
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()
    snippets_main_after_stop(tb)

if __name__ == '__main__':
    main()
