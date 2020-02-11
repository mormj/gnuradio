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

def get_nonblock_python(header_info, base_name, namespace, prefix_include_root):
    current_path = os.path.dirname(pathlib.Path(__file__).absolute())
    tpl = Template(filename=os.path.join(current_path,'pybind11_templates','license.mako'))
    license = str_to_fancyc_comment(tpl.render(year=datetime.now().year))

    tpl = Template(filename=os.path.join(current_path,'pybind11_templates','nonblock_python_hpp.mako'))
    return tpl.render(
        license=license,
        header_info=header_info,
        basename = base_name,
        # namespace = namespace
        prefix_include_root = prefix_include_root,
        module = True
    )

def write_bindings_generic(module_path, base_name, header_info, output_dir, namespace, prefix_include_root):
    json_pathname = os.path.join(output_dir,'{}.json'.format(base_name))
    binding_pathname = os.path.join(output_dir,'{}_python.hpp'.format(base_name))
    with open(json_pathname, 'w') as outfile:
        json.dump(header_info, outfile)

    try:
        pybind_code = get_nonblock_python(header_info, base_name, namespace, prefix_include_root)
        with open(binding_pathname, 'w+') as outfile:
            outfile.write(pybind_code)
        return binding_pathname
    except:
        return None

def process_nonblock_header_file(file_to_process, module_path, prefix, output_dir, namespace, prefix_include_root):
    binding_pathname = None
    # module_include_path = os.path.abspath(os.path.join(module_path, 'include'))
    base_name = os.path.splitext(os.path.basename(file_to_process))[0]
    module_include_path = os.path.abspath(os.path.dirname(module_path))
    # blocks_include_path=os.path.abspath(os.path.join(module_path,'..','gr-blocks','include'))
    # gr_include_path=os.path.abspath(os.path.join(module_path,'..','gnuradio-runtime','include'))
    # include_paths = ','.join((module_include_path,blocks_include_path,gr_include_path))
    prefix_include_path = os.path.abspath(os.path.join(prefix, 'include'))
    include_paths = ','.join((prefix_include_path, module_include_path))
    parser = GenericHeaderParser(
        include_paths=include_paths, file_path=file_to_process)
    try:
        header_info = parser.get_header_info(namespace)
        binding_pathname = write_bindings_generic(module_path, base_name, header_info, output_dir, namespace, prefix_include_root)
    except Exception as e:
        print(e)
        failure_pathname = os.path.join(output_dir,'failed_conversions.txt')
        with open(failure_pathname, 'a+') as outfile:
            outfile.write(file_to_process)
            outfile.write(str(e))
            outfile.write('\n')

    return binding_pathname

def process_file(args):

    file = args.filename
    module_path = pathlib.Path(file).absolute()  # path that the file lives in
    prefix = args.prefix

    # output_dir = os.path.join(args.output,  os.path.split(os.path.dirname(file)[1]))
    if 'include'+os.path.sep in file:
        rel_path_after_include = os.path.split(file.split('include'+os.path.sep)[-1])[0]
        output_dir = os.path.join(args.output, rel_path_after_include)
        # output_dir = os.path.join(output_dir,os.path.basename(file))
        if output_dir and not os.path.exists(output_dir):
            output_dir = os.path.abspath(output_dir)
            print('creating directory {}'.format(output_dir))
            os.makedirs(output_dir)

        return process_nonblock_header_file(file,
                    module_path, prefix, output_dir, args.namespace, args.prefix_include_root)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'filename',
        help='Path of gnuradio file to process')
    parser.add_argument(
        '--prefix',
        '-p',
        help='Path of gnuradio installation prefix')

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
    process_file(args)


if __name__ == "__main__":
    main()