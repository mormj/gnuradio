#!/usr/bin/env python3
# Copyright (C) 2020 Free Software Foundation
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

import os
from os import path
import pathlib
import re
from argparse import ArgumentParser
from gnuradio.blocktool import BlockHeaderParser, GenericHeaderParser
import json
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

def get_nonblock_python(header_info):
    current_path = os.path.dirname(pathlib.Path(__file__).absolute())
    tpl = Template(filename=os.path.join(current_path,'pybind11_templates','license.mako'))
    license = str_to_fancyc_comment(tpl.render(year=datetime.now().year))

    tpl = Template(filename=os.path.join(current_path,'pybind11_templates','nonblock_python_hpp.mako'))
    return tpl.render(
        license=license,
        header_info=header_info
    )


def get_block_python(header_info):
    current_path = os.path.dirname(pathlib.Path(__file__).absolute())
    tpl = Template(filename=os.path.join(current_path,'pybind11_templates','license.mako'))
    license = str_to_fancyc_comment(tpl.render(year=datetime.now().year))

    tpl = Template(filename=os.path.join(current_path,'pybind11_templates','block_python_hpp.mako'))
    return tpl.render(
        license=license,
        header_info=header_info
    )


def write_bindings_generic(module_name, module_path, header_info, output_dir):
    # python/[module_name]/bindings
    #  --> does [module_name]_python.hpp exist?

    #  --> create [module_name]_python.hpp

    # python/[module_name]
    #  --> Update CMakeLists.txt with Python Bindings info
    #  --> (if not there) create python_bindings.cpp
    #  --> add bind_[module_name] module to python_bindings.cpp

    json_pathname = os.path.join(output_dir,'{}.json'.format('::'.join(header_info['namespace'])))
    binding_pathname = os.path.join(output_dir,'{}_python.hpp'.format('::'.join(header_info['namespace'])))
    with open(json_pathname, 'w') as outfile:
        json.dump(header_info, outfile)

    try:
        pybind_code = get_nonblock_python(header_info)
        with open(binding_pathname, 'w+') as outfile:
            outfile.write(pybind_code)
    except:
        pass    

def write_bindings(module_name, module_path, header_info, output_dir):
    # python/[module_name]/bindings
    #  --> does [module_name]_python.hpp exist?

    #  --> create [module_name]_python.hpp

    # python/[module_name]
    #  --> Update CMakeLists.txt with Python Bindings info
    #  --> (if not there) create python_bindings.cpp
    #  --> add bind_[module_name] module to python_bindings.cpp

    json_pathname = os.path.join(output_dir,'{}.json'.format(header_info['class']))
    binding_pathname = os.path.join(output_dir,'{}_python.hpp'.format(header_info['class']))
    with open(json_pathname, 'w') as outfile:
        json.dump(header_info, outfile)

    try:
        pybind_code = get_block_python(header_info)
        with open(binding_pathname, 'w+') as outfile:
            outfile.write(pybind_code)
    except:
        pass

def process_header_file(file_to_process, module_name, module_path, prefix, output_dir):
    module_include_path = os.path.abspath(os.path.join(module_path, 'include'))
    # blocks_include_path=os.path.abspath(os.path.join(module_path,'..','gr-blocks','include'))
    # gr_include_path=os.path.abspath(os.path.join(module_path,'..','gnuradio-runtime','include'))
    # include_paths = ','.join((module_include_path,blocks_include_path,gr_include_path))
    prefix_include_path = os.path.abspath(os.path.join(prefix, 'include'))
    include_paths = ','.join((prefix_include_path, module_include_path))
    parser = BlockHeaderParser(
        include_paths=include_paths, file_path=file_to_process)
    try:
        header_info = parser.get_header_info()
        write_bindings(module_name, module_path, header_info, output_dir)
    except Exception as e:
        print(e)
        failure_pathname = os.path.join(output_dir,'failed_conversions.txt')
        with open(failure_pathname, 'w+') as outfile:
            outfile.write(file_to_process)
            outfile.write('\n')

def process_nonblock_header_file(file_to_process, module_name, module_path, prefix, output_dir, namespace):
    module_include_path = os.path.abspath(os.path.join(module_path, 'include'))
    # blocks_include_path=os.path.abspath(os.path.join(module_path,'..','gr-blocks','include'))
    # gr_include_path=os.path.abspath(os.path.join(module_path,'..','gnuradio-runtime','include'))
    # include_paths = ','.join((module_include_path,blocks_include_path,gr_include_path))
    prefix_include_path = os.path.abspath(os.path.join(prefix, 'include'))
    include_paths = ','.join((prefix_include_path, module_include_path))
    parser = GenericHeaderParser(
        include_paths=include_paths, file_path=file_to_process)
    try:
        header_info = parser.get_header_info(namespace)
        write_bindings_generic(module_name, module_path, header_info, output_dir)
    except Exception as e:
        print(e)
        failure_pathname = os.path.join(output_dir,'failed_conversions.txt')
        with open(failure_pathname, 'w+') as outfile:
            outfile.write(file_to_process)
            outfile.write('\n')


def process_module_files(args):

    module_path = args.module
    prefix = args.prefix
    module_name = os.path.basename(args.module)
    if module_name.startswith('gr-'):
        module_name = module_name.split('gr-')[1]

    output_dir = args.output
    output_dir = os.path.join(output_dir,module_name)
    if output_dir and not os.path.exists(output_dir):
        output_dir = os.path.abspath(output_dir)
        print('creating directory {}'.format(output_dir))
        os.mkdir(output_dir)

    if os.path.isdir(module_path):
        # drill down to the include directory
        for root, dirs, files in os.walk(os.path.join(module_path, 'include')):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            if path[-1] == module_name:
                for file in files:
                    print(len(path) * '---', file)
                    _, file_extension = os.path.splitext(file)
                    if (file_extension == '.h'):
                        process_header_file(os.path.join(root, file),
                                        module_name, module_path, prefix, output_dir)
    elif os.path.isfile(module_path):
        file = module_path
        _, file_extension = os.path.splitext(file)
        if (file_extension == '.h'):
            if args.nonblock:
                process_nonblock_header_file(file,
                            module_name, os.path.abspath(os.path.join(os.path.dirname(module_path),'..','..')), prefix, output_dir, args.namespace)
            else:
                process_header_file(file,
                            module_name, os.path.abspath(os.path.join(os.path.dirname(module_path),'..','..','..')), prefix, output_dir)



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--prefix',
        '-p',
        help='Path of gnuradio installation prefix')
    parser.add_argument(
        '--module',
        '-m',
        help='Path of gnuradio module to process')
    parser.add_argument(
        '--nonblock',
        '-n',
        action='store_true',
        help='Flag to process a non block header file'
    )
    parser.add_argument(
        '--namespace',
        nargs='+',
        help='namespace to parse',
        default=[]
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Path to output directory for generated files (temporary)',
        default = '')
    return parser.parse_args()


def main():
    """
    Run this if the program was invoked on the commandline
    """
    args = parse_args()
    process_module_files(args)


if __name__ == "__main__":
    exit(not (main()))
