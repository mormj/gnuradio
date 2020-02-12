import argparse
import os
from gnuradio.bindtool import BindingGenerator
import pathlib

parser = argparse.ArgumentParser(description='Bind a GR In-Tree Module')
parser.add_argument('names', type=str, nargs='+',
                    help='Names of gr modules to bind (e.g. fft digital analog)')

parser.add_argument('--output_dir', default = '/tmp',
                    help='Output directory of generated bindings')
parser.add_argument('--prefix', help='Prefix of Installed GNU Radio')
parser.add_argument('--src', help='Directory of gnuradio source tree', default=os.path.dirname(os.path.abspath(__file__))+'/../../../..')
args = parser.parse_args()


print(pathlib.Path(__file__).parent.absolute())
print(args)

prefix = args.prefix
output_dir = args.output_dir
for name in args.names:
    namespace = ['gr',name]
    module_dir = os.path.join(args.src,'gr-'+name,'include')
    prefix_include_root = 'gnuradio/'+name  #pmt, gnuradio/digital, etc.

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bg = BindingGenerator()
        bg.gen_bindings(module_dir, prefix, namespace, prefix_include_root, output_dir)