#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: josh
# GNU Radio version: 3.8.0.0


from gnuradio import gr
from gnuradio import blocks
import sys
import signal
import tempfile
import array

class simplest(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################
        _datafile = tempfile.NamedTemporaryFile()
        _datafilename = _datafile.name
        _vector = [x for x in range(1000)]

        with open(_datafilename, 'wb') as f:
            array.array('f', _vector).tofile(f)

        src = blocks.file_source(gr.sizeof_float, _datafilename)
        snk = blocks.vector_sink_f()
        mc = blocks.multiply_const_ff(3)

        mc.set_k(7)
        mc.k()
        # b = mc.to_bb()
        # b = mc.to_basic_block()
        # b = ns.to_basic_block()
        # b = vs.to_basic_block()
        

        ##################################################
        # Connections
        ##################################################
        self.connect(src, mc, snk)

        self.snk = snk

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate



def main(top_block_cls=simplest, options=None):
    tb = top_block_cls()

    # def sig_handler(sig=None, frame=None):
    #     tb.stop()
    #     tb.wait()
    #     sys.exit(0)

    # signal.signal(signal.SIGINT, sig_handler)
    # signal.signal(signal.SIGTERM, sig_handler)

    # print(tb.vsi.data())
    # tb.start()
    # try:
    #     input('Press Enter to quit: ')
    # except EOFError:
    #     pass
    # tb.stop()
    # tb.wait()
    
    tb.run()
    print(tb.snk.data())


if __name__ == '__main__':
    main()
