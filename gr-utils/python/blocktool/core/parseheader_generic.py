
## TODO: Make this process any header file, non necessarily a block, e.g. pmt.h
#
# Copyright 2019 Free Software Foundation, Inc.
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
#
""" Module to generate AST for the headers and parse it """

from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import re
import codecs
import logging

from pygccxml import parser, declarations, utils

from ..core.base import BlockToolException, BlockTool
from ..core.iosignature import io_signature, message_port
from ..core.comments import read_comments, add_comments, exist_comments
from ..core import Constants

LOGGER = logging.getLogger(__name__)


class GenericHeaderParser(BlockTool):
    """
    : Single argument required: file_path
    file_path: enter path for the header block in any of GNU Radio module
        : returns the parsed header data in python dict
        : return dict keys: namespace, class, io_signature, make,
                       properties, methods
    : Can be used as an CLI command or an extenal API
    """
    name = 'Block Parse Header'
    description = 'Create a parsed output from a block header file'

    def __init__(self, file_path=None, blocktool_comments=False, include_paths=None, **kwargs):
        """ __init__ """
        BlockTool.__init__(self, **kwargs)
        self.parsed_data = {}
        self.addcomments = blocktool_comments
        self.include_paths = None
        if (include_paths):
            self.include_paths = [p.strip() for p in include_paths.split(',')]
        if not os.path.isfile(file_path):
            raise BlockToolException('file does not exist')
        file_path = os.path.abspath(file_path)
        self.target_file = file_path
        self.initialize()
        self.validate()

    def initialize(self):
        """
        initialize all the required API variables
        """
        self.module = self.target_file
        for dirs in self.module:
            if not os.path.basename(self.module).startswith(Constants.GR):
                self.module = os.path.abspath(
                    os.path.join(self.module, os.pardir))
        self.modname = os.path.basename(self.module)
        self.filename = os.path.basename(self.target_file)
        self.targetdir = os.path.dirname(self.target_file)

    def validate(self):
        """ Override the Blocktool validate function """
        BlockTool._validate(self)
        if not self.filename.endswith('.h'):
            raise BlockToolException(
                'Cannot parse a non-header file')

    def get_header_info(self, namespace_to_parse):
        """
        PyGCCXML header code parser
        magic happens here!
        : returns the parsed header data in python dict
        : return dict keys: namespace, class, io_signature, make,
                       properties, methods
        : Can be used as an CLI command or an extenal API
        """
        gr = self.modname.split('-')[0]
        module = self.modname.split('-')[-1]
        self.parsed_data['module_name'] = module
        generator_path, generator_name = utils.find_xml_generator()
        xml_generator_config = parser.xml_generator_configuration_t(
            xml_generator_path=generator_path,
            xml_generator=generator_name,
            include_paths=self.include_paths,
            compiler='gcc',
            define_symbols=['BOOST_ATOMIC_DETAIL_EXTRA_BACKEND_GENERIC'],
            cflags='-std=c++11')
        decls = parser.parse(
            [self.target_file], xml_generator_config)
        global_namespace = declarations.get_global_namespace(decls)


        # namespace
        try:
            self.parsed_data['namespace'] = []
            main_namespace = global_namespace
            for ns in namespace_to_parse:
                main_namespace = main_namespace.namespace(ns)
            if main_namespace is None:
                raise BlockToolException('namespace cannot be none')
            self.parsed_data['namespace'] = namespace_to_parse

        except RuntimeError:
            raise BlockToolException(
                'Invalid namespace format in the block header file')

        # enums
        try:
            self.parsed_data['enums'] = []
            enums = main_namespace.enumerations(header_file=self.target_file)
            if enums:
                for _enum in enums:
                    current_enum = {'name': _enum.name, 'values':_enum.values}
                    self.parsed_data['enums'].append(current_enum)
        except:
            pass

        # variables
        try:
            self.parsed_data['variables'] = []
            variables = main_namespace.variables(header_file=self.target_file)
            if variables:
                for _var in variables:
                    current_var = {'name': _var.name, 'values':_var.value}
                    self.parsed_data['vars'].append(current_var)
        except:
            pass

        # classes
        try:
            self.parsed_data['classes'] = []

            # query_methods = declarations.access_type_matcher_t('public')
            classes = main_namespace.classes(header_file=self.target_file)
            if classes:
                for _class in classes:
            # for _class in main_namespace.declarations:
                # if isinstance(_class, declarations.class_t):
                    current_class = {'name': _class.name, 'member_functions':[]}
                    if _class.bases:
                        current_class['bases'] = _class.bases[0].declaration_path
                    member_functions = []
                    constructors = []                    
                    # constructors
                    try: 
                        query_methods = declarations.access_type_matcher_t('public')
                        cotrs = _class.constructors(function=query_methods,
                                                            allow_empty=True,
                                                            header_file=self.target_file,
                                                            name=_class.name)
                        for cotr in cotrs:
                            
                            cotr_args = {
                                "name": str(cotr.name),
                                "arguments": []
                            }
                            for argument in cotr.arguments:
                                args = {
                                    "name": str(argument.name),
                                    "dtype": str(argument.decl_type),
                                    "default": argument.default_value
                                }
                                cotr_args['arguments'].append(args.copy())
                            constructors.append(cotr_args.copy())
                        current_class['constructors'] = constructors
                    except RuntimeError:
                        pass

                    # class member functions
                    try:
                        
                        query_methods = declarations.access_type_matcher_t('public')
                        functions = _class.member_functions(function=query_methods,
                                                            allow_empty=True,
                                                            header_file=self.target_file)
                        for fcn in functions:
                            if str(fcn.name) not in [_class.name, '~'+_class.name]:
                                fcn_args = {
                                    "name": str(fcn.name),
                                    "return_type": str(fcn.return_type),
                                    "arguments": []
                                }
                                for argument in fcn.arguments:
                                    args = {
                                        "name": str(argument.name),
                                        "dtype": str(argument.decl_type),
                                        "default": argument.default_value
                                        
                                    }
                                    fcn_args['arguments'].append(args.copy())
                                member_functions.append(fcn_args.copy())
                        current_class['member_functions'] = member_functions
                    except RuntimeError:
                        pass

                    # enums
                    try:
                        class_enums = []
                        enums = _class.enumerations(header_file=self.target_file)
                        if enums:
                            for _enum in enums:
                                current_enum = {'name': _enum.name, 'values':_enum.values}
                                class_enums.appen(current_enum)
                        
                        current_class['enums'] = class_enums
                    except:
                        pass

                    # variables
                    try:
                        class_vars = []
                        query_methods = declarations.access_type_matcher_t('public')
                        variables = _class.variables(unction=query_methods,
                            header_file=self.target_file)
                        if variables:
                            for _var in variables:
                                current_var = {'name': _var.name, 'value':_var.value}
                                class_vars.append(current_var)
                        current_class['vars'] = class_vars

                    except:
                        pass

                    self.parsed_data['classes'].append(current_class)


        except:
            pass

                    

        # free functions
        try:
            self.parsed_data['free_functions'] = []
            free_functions = []
            functions = main_namespace.free_functions(allow_empty=True,
                                                header_file=self.target_file)
            for fcn in functions:
                if str(fcn.name) not in ['make']:
                    fcn_args = {
                        "name": str(fcn.name),
                        "return_type": str(fcn.return_type),
                        "arguments": []
                    }
                    for argument in fcn.arguments:
                        args = {
                            "name": str(argument.name),
                            "dtype": str(argument.decl_type),
                            "default": argument.default_value
                        }
                        fcn_args['arguments'].append(args.copy())
                    free_functions.append(fcn_args.copy())
   
            self.parsed_data['free_functions'] = free_functions
        except RuntimeError:
            pass
        # namespace

        return self.parsed_data

    def run_blocktool(self):
        """ Run, run, run. """
        pass
        # self.get_header_info()
