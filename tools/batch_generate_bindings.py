#!/usr/bin/env python3

import gen_nonblock_bindings
import os
import argparse




def get_file_list(include_path):
    file_list = []
    for root, dirs, files in os.walk(include_path):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            print(len(path) * '---', file)
            _, file_extension = os.path.splitext(file)
            if (file_extension == '.h'):
                pathname = os.path.join(root, file)
                print(pathname)
                file_list.append(pathname)
    return file_list


def gen_bindings(file_list, output_dir, prefix, namespace, prefix_include_root):
    file_list = get_file_list(include_path)
    for fn in file_list:
        args = argparse.Namespace(filename=fn, output=output_dir,
                                  prefix=prefix, namespace=namespace,
                                  prefix_include_root=prefix_include_root)
        gen_nonblock_bindings.process_file(args)

prefix = '/share/gnuradio/grnext'
include_path = '/share/gnuradio/grnext/src/gnuradio/gnuradio-runtime/include/gnuradio'
output_dir = '/share/tmp/blocktool_pybind'
namespace = ['gr']
bindings_path = '/share/gnuradio/grnext/src/gnuradio/gnuradio-runtime/python/gnuradio/gr/bindings/tmp'
prefix_include_root = 'gnuradio'  #pmt, gnuradio/digital, etc.

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    gen_bindings(get_file_list(include_path), output_dir, prefix, namespace, prefix_include_root)