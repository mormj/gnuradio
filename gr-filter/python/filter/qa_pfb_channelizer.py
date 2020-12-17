#!/usr/bin/env python
#
# Copyright 2012-2014 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
#


from gnuradio import gr, gr_unittest, filter, blocks, analog
import math
import cmath


def sig_source_c(samp_rate, freq, amp, N):
    t = [float(x) / samp_rate for x in range(N)]
    y = [math.cos(2. * math.pi * freq * x) +
         1j * math.sin(2. * math.pi * freq * x) for x in t]
    return y


class test_pfb_channelizer(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()
        self.freqs = [110., -513., 203., -230, 121]
        # Number of channels to channelize.
        self.M = len(self.freqs)
        # Number of samples to use.
        self.N = 1000
        # Baseband sampling rate.
        self.fs = 5000
        # Input samp rate to channelizer.
        self.ifs = self.M * self.fs

        self.taps = filter.firdes.low_pass_2(
            1, self.ifs, self.fs / 2, self.fs / 10,
            attenuation_dB=80,
            window=fft.window.WIN_BLACKMAN_hARRIS)

        self.Ntest = 50

    def tearDown(self):
        self.tb = None

    def test_0000(self):
        self.check_channelizer(filter.pfb.channelizer_ccf(
            self.M, taps=self.taps, oversample_rate=1))

    def test_0001(self):
        self.check_channelizer(filter.pfb.channelizer_hier_ccf(
            self.M, n_filterbanks=1, taps=self.taps))

    def test_0002(self):
        """Test roundig error handling for oversample rate (ok)."""
        channels, oversample = 36, 25.
        filter.pfb.channelizer_ccf(channels, taps=self.taps,
                                   oversample_rate=channels / oversample)

    def test_0003(self):
        """Test roundig error handling for oversample rate, (bad)."""
        # pybind11 raises ValueError instead of TypeError
        self.assertRaises(ValueError,
                          filter.pfb.channelizer_ccf,
                          36, taps=self.taps, oversample_rate=10.1334)

    def get_input_data(self):
        """
        Get the raw data generated by addition of sinusoids.
        Useful for debugging.
        """
        tb = gr.top_block()
        signals = []
        add = blocks.add_cc()
        for i in range(len(self.freqs)):
            f = self.freqs[i] + i * self.fs
            signals.append(
                analog.sig_source_c(
                    self.ifs,
                    analog.GR_SIN_WAVE,
                    f,
                    1))
            tb.connect(signals[i], (add, i))
        head = blocks.head(gr.sizeof_gr_complex, self.N)
        snk = blocks.vector_sink_c()
        tb.connect(add, head, snk)
        tb.run()
        input_data = snk.data()
        return input_data

    def check_channelizer(self, channelizer_block):
        signals = list()
        add = blocks.add_cc()
        for i in range(len(self.freqs)):
            f = self.freqs[i] + i * self.fs
            data = sig_source_c(self.ifs, f, 1, self.N)
            signals.append(blocks.vector_source_c(data))
            self.tb.connect(signals[i], (add, i))

        #s2ss = blocks.stream_to_streams(gr.sizeof_gr_complex, self.M)

        #self.tb.connect(add, s2ss)
        self.tb.connect(add, channelizer_block)

        snks = list()
        for i in range(self.M):
            snks.append(blocks.vector_sink_c())
            #self.tb.connect((s2ss,i), (channelizer_block,i))
            self.tb.connect((channelizer_block, i), snks[i])

        self.tb.run()

        L = len(snks[0].data())

        expected_data = self.get_expected_data(L)
        received_data = [snk.data() for snk in snks]

        for expected, received in zip(expected_data, received_data):
            self.compare_data(expected, received)

    def compare_data(self, expected, received):
        Ntest = 50
        expected = expected[-Ntest:]
        received = received[-Ntest:]
        expected = [x / expected[0] for x in expected]
        received = [x / received[0] for x in received]
        self.assertComplexTuplesAlmostEqual(expected, received, 3)

    def get_freq(self, data):
        freqs = []
        for r1, r2 in zip(data[:-1], data[1:]):
            diff = cmath.phase(r1) - cmath.phase(r2)
            if diff > math.pi:
                diff -= 2 * math.pi
            if diff < -math.pi:
                diff += 2 * math.pi
            freqs.append(diff)
        freq = float(sum(freqs)) / len(freqs)
        freq /= 2 * math.pi
        return freq

    def get_expected_data(self, L):

        # Filter delay is the normal delay of each arm
        tpf = math.ceil(len(self.taps) / float(self.M))
        delay = -(tpf - 1.0) / 2.0
        delay = int(delay)

        # Create a time scale that's delayed to match the filter delay
        t = [float(x) / self.fs for x in range(delay, L + delay)]

        # Create known data as complex sinusoids at the different baseband freqs
        # the different channel numbering is due to channelizer output order.
        expected_data = [[math.cos(2. *
                                   math.pi *
                                   f *
                                   x) +
                          1j *
                          math.sin(2. *
                                   math.pi *
                                   f *
                                   x) for x in t] for f in self.freqs]
        return expected_data


if __name__ == '__main__':
    gr_unittest.run(test_pfb_channelizer)
