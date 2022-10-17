#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: josh
# GNU Radio version: 4.0.0.0-preview0

from gnuradio import analog
from gnuradio import blocks
from gnuradio import fileio
from gnuradio import filter
from gnuradio import gr
#from gnuradio.filter import firdes
#from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
#from gnuradio.eng_arg import eng_float, intx
#from gnuradio import eng_notation
from gnuradio import math
from gnuradio import streamops
from gnuradio.kernel.fft import window
from gnuradio.kernel.filter import firdes
import math as pmath



def snipfcn_snippet_0(fg, rt=None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.plot(np.real(fg.snk.data()))
    plt.plot(np.imag(fg.snk.data()))
    plt.figure()
    plt.plot(np.real(fg.snk_0.data()))

    plt.plot(np.imag(fg.snk_0.data()))
    plt.show()


def snippets_main_after_stop(fg, rt=None):
    snipfcn_snippet_0(fg, rt)


class debug_wfm_pll(gr.flowgraph):

    def __init__(self):
        gr.flowgraph.__init__(self, "Not titled yet")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 768000
        self.rf_decim = rf_decim = 2
        self.demod_rate = demod_rate = (int)(samp_rate/rf_decim)
        self.stereo_carrier_filter_coeffs = stereo_carrier_filter_coeffs = firdes.band_pass( -2.0, demod_rate, 37600, 38400, 400, window.HAMMING, 6.76)
        self.pilot_carrier_filter_coeffs = pilot_carrier_filter_coeffs = firdes.complex_band_pass( 1.0, demod_rate, 18980, 19020, 1500, window.HAMMING, 6.76)
        self.audio_decim = audio_decim = (int)(demod_rate/48000)
        self.samp_delay = samp_delay = (len( pilot_carrier_filter_coeffs) - 1) // 2 + (len(stereo_carrier_filter_coeffs) - 1) // 2
        self.deviation = deviation = 75000
        self.deemph_tau = deemph_tau = 75e-6
        self.audio_rate = audio_rate = demod_rate / audio_decim
        self.audio_filter_coeffs = audio_filter_coeffs = firdes.low_pass(1, demod_rate, 15000, 1500, window.HAMMING, 6.76)

        ##################################################
        # Blocks
        ##################################################
        self.streamops_head_0_0 = streamops.head( 10000,0, impl=streamops.head.cpu)
        self.streamops_head_0 = streamops.head( 10000,0, impl=streamops.head.cpu)
        self.streamops_delay_0 = streamops.delay( samp_delay,gr.sizeof_float, impl=streamops.delay.cpu)
        self.stereo_carrier_bpf = filter.fft_filter_fff( 1,stereo_carrier_filter_coeffs,0,0, impl=filter.fft_filter_fff.cpu)
        self.stereo_audio_lpf = filter.fft_filter_fff( 1,audio_filter_coeffs,0,0, impl=filter.fft_filter_fff.cpu)
        self.snk_0 = blocks.vector_sink_f( 1,1024, impl=blocks.vector_sink_f.cpu)
        self.snk = blocks.vector_sink_f( 1,1024, impl=blocks.vector_sink_f.cpu)
        self.pilot_carrier_bpf = filter.fir_filter_fcc( 1,pilot_carrier_filter_coeffs, impl=filter.fir_filter_fcc.cpu)
        self.mono_audio_lpf = filter.fft_filter_fff( 1,audio_filter_coeffs,0,0, impl=filter.fft_filter_fff.cpu)
        self.math_sub_0 = math.sub_ff( 2,1, impl=math.sub_ff.cpu)
        self.math_complex_to_imag_0 = math.complex_to_imag( 1, impl=math.complex_to_imag.cpu)
        self.math_add_0 = math.add_ff( 2,1, impl=math.add_ff.cpu)
        self.fileio_file_source_0 = fileio.file_source( '/tmp/fm_in.dat',False,0,0,0, impl=fileio.file_source.cpu)
        self.blocks_stereo_multiply = math.multiply_ff( 2,1, impl=math.multiply_ff.cpu)
        self.blocks_pilot_multiply = math.multiply_cc( 2,1, impl=math.multiply_cc.cpu)
        self.blocks_null_sink_0_0 = blocks.null_sink( 1,0, impl=blocks.null_sink.cpu)
        self.blocks_null_sink_0 = blocks.null_sink( 1,0, impl=blocks.null_sink.cpu)
        self.analog_right_fm_deemph = analog.fm_deemph( 400000,75e-6, impl=analog.fm_deemph.cpu)
        self.analog_quadrature_demod_0 = analog.quadrature_demod( demod_rate / (2 * pmath.pi * deviation), impl=analog.quadrature_demod.cpu)
        self.analog_pll_refout_0 = analog.pll_refout( 0.001,2 * pmath.pi * 19200 / demod_rate,2 * pmath.pi * 18800 / demod_rate, impl=analog.pll_refout.cpu)
        self.analog_left_fm_deemph = analog.fm_deemph( 400000,75e-6, impl=analog.fm_deemph.cpu)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_left_fm_deemph, 0), (self.blocks_null_sink_0, 0))
        self.connect((self.analog_left_fm_deemph, 0), (self.streamops_head_0, 0))
        self.connect((self.analog_pll_refout_0, 0), (self.blocks_pilot_multiply, 0))
        self.connect((self.analog_pll_refout_0, 0), (self.blocks_pilot_multiply, 1))
        self.connect((self.analog_quadrature_demod_0, 0), (self.pilot_carrier_bpf, 0))
        self.connect((self.analog_quadrature_demod_0, 0), (self.streamops_delay_0, 0))
        self.connect((self.analog_right_fm_deemph, 0), (self.blocks_null_sink_0_0, 0))
        self.connect((self.analog_right_fm_deemph, 0), (self.streamops_head_0_0, 0))
        self.connect((self.blocks_pilot_multiply, 0), (self.math_complex_to_imag_0, 0))
        self.connect((self.blocks_stereo_multiply, 0), (self.stereo_audio_lpf, 0))
        self.connect((self.fileio_file_source_0, 0), (self.analog_quadrature_demod_0, 0))
        self.connect((self.math_add_0, 0), (self.analog_left_fm_deemph, 0))
        self.connect((self.math_complex_to_imag_0, 0), (self.stereo_carrier_bpf, 0))
        self.connect((self.math_sub_0, 0), (self.analog_right_fm_deemph, 0))
        self.connect((self.mono_audio_lpf, 0), (self.math_add_0, 1))
        self.connect((self.mono_audio_lpf, 0), (self.math_sub_0, 0))
        self.connect((self.pilot_carrier_bpf, 0), (self.analog_pll_refout_0, 0))
        self.connect((self.stereo_audio_lpf, 0), (self.math_add_0, 0))
        self.connect((self.stereo_audio_lpf, 0), (self.math_sub_0, 1))
        self.connect((self.stereo_carrier_bpf, 0), (self.blocks_stereo_multiply, 0))
        self.connect((self.streamops_delay_0, 0), (self.blocks_stereo_multiply, 1))
        self.connect((self.streamops_delay_0, 0), (self.mono_audio_lpf, 0))
        self.connect((self.streamops_head_0, 0), (self.snk, 0))
        self.connect((self.streamops_head_0_0, 0), (self.snk_0, 0))


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
        self.set_audio_filter_coeffs(firdes.low_pass(1, self.demod_rate, 15000, 1500, window.HAMMING, 6.76))
        self.set_audio_rate(self.demod_rate / self.audio_decim)
        self.set_pilot_carrier_filter_coeffs(firdes.complex_band_pass( 1.0, self.demod_rate, 18980, 19020, 1500, window.HAMMING, 6.76))
        self.set_stereo_carrier_filter_coeffs(firdes.band_pass( -2.0, self.demod_rate, 37600, 38400, 400, window.HAMMING, 6.76))

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




def main(flowgraph_cls=debug_wfm_pll, options=None):
    fg = flowgraph_cls()
    rt = gr.runtime()


    rt.initialize(fg)

    rt.start()

    try:
        rt.wait()
    except KeyboardInterrupt:
        rt.stop()
        rt.wait()
    snippets_main_after_stop(fg, rt)

if __name__ == '__main__':
    main()
