from gnuradio.bindtool import BindingGenerator

prefix = '/share/gnuradio/grnext'
module_dir = '/share/gnuradio/grnext/src/gnuradio/gnuradio-runtime/include/gnuradio'
output_dir = '/share/tmp/test_pybind'
namespace = ['gr']
bindings_path = '/share/gnuradio/grnext/src/gnuradio/gnuradio-runtime/python/gnuradio/gr/bindings/tmp'
prefix_include_root = 'gnuradio'  #pmt, gnuradio/digital, etc.

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    bg = BindingGenerator()
    bg.gen_bindings(module_dir, prefix, namespace, prefix_include_root, output_dir)