#!/usr/bin/env python3

import gen_nonblock_bindings
import os
import argparse
import os
from os import path
import pathlib
import re
from mako.template import Template
from datetime import datetime

def str_to_fancyc_comment(text):
    """ Return a string as a C formatted comment. """
    l_lines = text.splitlines()
    if len(l_lines[0]) == 0:
        outstr = "/*\n"
    else:
        outstr = "/* " + l_lines[0] + "\n"
    for line in l_lines[1:]:
        if len(line) == 0:
            outstr += " *\n"
        else:
            outstr += " * " + line + "\n"
    outstr += " */\n"
    return outstr



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
    # file_list = get_file_list(include_path)
    for fn in file_list:
        args = argparse.Namespace(filename=fn, output=output_dir,
                                  prefix=prefix, namespace=namespace,
                                  prefix_include_root=prefix_include_root)
        gen_nonblock_bindings.process_file(args)

def gen_top_level_cpp(file_list, output_dir):
    

    current_path = os.path.dirname(pathlib.Path(__file__).absolute())

    file = file_list[0]
    
    if 'include'+os.path.sep in file:
        rel_path_after_include = os.path.split(file.split('include'+os.path.sep)[-1])[0]
        output_dir = os.path.join(output_dir, rel_path_after_include)
        # output_dir = os.path.join(output_dir,os.path.basename(file))
        if output_dir and not os.path.exists(output_dir):
            output_dir = os.path.abspath(output_dir)
            print('creating directory {}'.format(output_dir))
            os.makedirs(output_dir)

    tpl = Template(filename=os.path.join(current_path,'pybind11_templates','license.mako'))
    license = str_to_fancyc_comment(tpl.render(year=datetime.now().year))

    binding_pathname = os.path.join(output_dir,'python_bindings.cpp')
    file_list = [os.path.split(f)[-1] for f in file_list]
    tpl = Template(filename=os.path.join(current_path,'pybind11_templates','python_bindings_cpp.mako'))
    pybind_code = tpl.render(
        license=license,
        # namespace = namespace
        files = file_list
    )

    # print(pybind_code)
    try:
        with open(binding_pathname, 'w+') as outfile:
            outfile.write(pybind_code)
        return binding_pathname
    except:
        return None


prefix = '/share/gnuradio/grnext'
include_path = '/share/gnuradio/grnext/src/gnuradio/gnuradio-runtime/include/gnuradio'
output_dir = '/share/tmp/blocktool_pybind'
namespace = ['gr']
bindings_path = '/share/gnuradio/grnext/src/gnuradio/gnuradio-runtime/python/gnuradio/gr/bindings/tmp'
prefix_include_root = 'gnuradio'  #pmt, gnuradio/digital, etc.


prefix = '/share/gnuradio/grnext'
include_path = '/share/gnuradio/grnext/src/gnuradio/gr-blocks/include'
output_dir = '/share/tmp/blocktool_pybind'
namespace = ['gr','blocks']
prefix_include_root = 'gnuradio/blocks'  #pmt, gnuradio/digital, etc.

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    file_list = get_file_list(include_path)
    gen_top_level_cpp(file_list, output_dir)
    # gen_bindings(file_list, output_dir, prefix, namespace, prefix_include_root)