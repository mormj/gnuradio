#!/usr/bin/env python3

from gnuradio import gr_unittest, gr, blocks, streamops

class test_stream_adapters(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.flowgraph()

    def tearDown(self):
        self.tb = None

    def test_stream_adapters(self):
        nsamples = 100000
        input_data = list(range(nsamples))

        src = blocks.vector_source_f(input_data, False, vlen=10)
        cp1 = streamops.copy(gr.sizeof_float)
        snk1 = blocks.vector_sink_f(vlen=4)

        self.tb.connect(src, 0, cp1, 0)
        self.tb.connect(cp1, 0, snk1, 0)

        self.tb.run()
        
        self.assertEqual(input_data, snk1.data())

if __name__ == "__main__":
    gr_unittest.run(test_stream_adapters)