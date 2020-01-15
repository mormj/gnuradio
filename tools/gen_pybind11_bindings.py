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
import re
from argparse import ArgumentParser
from gnuradio.blocktool import BlockHeaderParser
import json


def write_bindings(module_name, module_path, header_info):
    # python/[module_name]/bindings
    #  --> does [module_name]_python.hpp exist?

    #  --> create [module_name]_python.hpp

    # python/[module_name]
    #  --> Update CMakeLists.txt with Python Bindings info
    #  --> (if not there) create python_bindings.cpp
    #  --> add bind_[module_name] module to python_bindings.cpp

    with open('{}.json'.format(header_info['class']), 'w') as outfile:
        json.dump(header_info, outfile)

    pass


def process_header_file(file_to_process, module_name, module_path, prefix):
    module_include_path = os.path.abspath(os.path.join(module_path, 'include'))
    # blocks_include_path=os.path.abspath(os.path.join(module_path,'..','gr-blocks','include'))
    # gr_include_path=os.path.abspath(os.path.join(module_path,'..','gnuradio-runtime','include'))
    # include_paths = ','.join((module_include_path,blocks_include_path,gr_include_path))
    prefix_include_path = os.path.abspath(os.path.join(prefix, 'include'))
    include_paths = ','.join((prefix_include_path, module_include_path))
    parser = BlockHeaderParser(
        include_paths=include_paths, file_path=file_to_process)
    header_info = parser.get_header_info()
    write_bindings(module_name, module_path, header_info)


def process_module_files(args):
    module_path = args.module
    prefix = args.prefix
    module_name = os.path.basename(args.module)
    if module_name.startswith('gr-'):
        module_name = module_name.split('gr-')[1]

    # drill down to the include directory
    for root, dirs, files in os.walk(os.path.join(module_path, 'include')):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        if path[-1] == module_name:
            for file in files:
                print(len(path) * '---', file)
                process_header_file(os.path.join(root, file),
                                    module_name, module_path, prefix)

        # self.module = self.target_file
        # for dirs in self.module:
        #     if not os.path.basename(self.module).startswith(Constants.GR):
        #         self.module = os.path.abspath(
        #             os.path.join(self.module, os.pardir))
        # self.modname = os.path.basename(self.module)
        # self.filename = os.path.basename(self.target_file)
        # self.targetdir = os.path.dirname(self.target_file)
        # for dirs in os.scandir(self.module):
        #     if dirs.is_dir():
        #         if dirs.path.endswith('lib'):
        #             self.impldir = dirs.path
        # self.impl_file = os.path.join(self.impldir,
        #                               self.filename.split('.')[0]+'_impl.cc')


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
    return parser.parse_args()


def main():
    """
    Run this if the program was invoked on the commandline
    """
    args = parse_args()

    process_module_files(args)


if __name__ == "__main__":
    exit(not (main()))
