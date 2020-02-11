from gnuradio.bindtool import BindingGenerator

prefix = '/share/gnuradio/grnext'
output_dir = '/share/tmp/test_pybind'
namespace = ['gr']
module_dir = '/share/gnuradio/grnext/src/gnuradio/gnuradio-runtime/include/gnuradio'
prefix_include_root = 'gnuradio'  #pmt, gnuradio/digital, etc.

prefix = '/share/gnuradio/grnext'
output_dir = '/share/tmp/test_pybind'
namespace = ['gr','digital']
module_dir = '/share/gnuradio/grnext/src/gnuradio/gr-digital/include/gnuradio/digital'
prefix_include_root = 'gnuradio/digital'  #pmt, gnuradio/digital, etc.

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    bg = BindingGenerator()
    bg.gen_bindings(module_dir, prefix, namespace, prefix_include_root, output_dir)