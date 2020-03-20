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

    def parse_function(self, func_decl):
        fcn_dict = {
            "name": str(func_decl.name),
            "return_type": str(func_decl.return_type),
            "has_static": func_decl.has_static if hasattr(func_decl, 'has_static') else '0',
            "arguments": []
        }
        for argument in func_decl.arguments:
            args = {
                "name": str(argument.name),
                "dtype": str(argument.decl_type),
                "default": argument.default_value
            }
            fcn_dict['arguments'].append(args)

        return fcn_dict

    def parse_class(self, class_decl):
        class_dict = {'name': class_decl.name, 'member_functions': []}
        if class_decl.bases:
            class_dict['bases'] = class_decl.bases[0].declaration_path

        constructors = []
        # constructors
        constructors = []
        query_methods = declarations.access_type_matcher_t('public')
        if hasattr(class_decl, 'constructors'):
            cotrs = class_decl.constructors(function=query_methods,
                                            allow_empty=True, recursive=False,
                                            header_file=self.target_file,
                                            name=class_decl.name)
            for cotr in cotrs:
                constructors.append(self.parse_function(cotr))

        class_dict['constructors'] = constructors

        # class member functions
        member_functions = []
        query_methods = declarations.access_type_matcher_t('public')
        if hasattr(class_decl, 'member_functions'):
            functions = class_decl.member_functions(function=query_methods,
                                                    allow_empty=True, recursive=False,
                                                    header_file=self.target_file)
            for fcn in functions:
                if str(fcn.name) not in [class_decl.name, '~'+class_decl.name]:
                    member_functions.append(self.parse_function(fcn))

        class_dict['member_functions'] = member_functions

        # enums
        class_enums = []
        if hasattr(class_decl, 'variables'):
            enums = class_decl.enumerations(
                allow_empty=True, recursive=False, header_file=self.target_file)
            if enums:
                for _enum in enums:
                    current_enum = {'name': _enum.name, 'values': _enum.values}
                    class_enums.append(current_enum)

        class_dict['enums'] = class_enums

        # variables
        class_vars = []
        query_methods = declarations.access_type_matcher_t('public')
        if hasattr(class_decl, 'variables'):
            variables = class_decl.variables(allow_empty=True, recursive=False, function=query_methods,
                                             header_file=self.target_file)
            if variables:
                for _var in variables:
                    current_var = {
                        'name': _var.name, 'value': _var.value, "has_static": _var.has_static}
                    class_vars.append(current_var)
        class_dict['vars'] = class_vars

        return class_dict

    def parse_namespace(self, namespace_decl):
        namespace_dict = {}
        # enums
        namespace_dict['name'] = namespace_decl.name
        namespace_dict['enums'] = []
        if hasattr(namespace_decl, 'enumerations'):
            enums = namespace_decl.enumerations(
                allow_empty=True, recursive=False, header_file=self.target_file)
            if enums:
                for _enum in enums:
                    current_enum = {'name': _enum.name, 'values': _enum.values}
                    namespace_dict['enums'].append(current_enum)

        # variables
        namespace_dict['variables'] = []
        if hasattr(namespace_decl, 'variables'):
            variables = namespace_decl.variables(
                allow_empty=True, recursive=False, header_file=self.target_file)
            if variables:
                for _var in variables:
                    current_var = {
                        'name': _var.name, 'values': _var.value, 'has_static': _var.has_static}
                    namespace_dict['vars'].append(current_var)

        # classes
        namespace_dict['classes'] = []
        if hasattr(namespace_decl, 'classes'):
            classes = namespace_decl.classes(
                allow_empty=True, recursive=False, header_file=self.target_file)
            if classes:
                for _class in classes:
                    namespace_dict['classes'].append(self.parse_class(_class))

        # free functions
        namespace_dict['free_functions'] = []
        free_functions = []
        if hasattr(namespace_decl, 'free_functions'):
            functions = namespace_decl.free_functions(allow_empty=True, recursive=False,
                                                      header_file=self.target_file)
            for fcn in functions:
                if str(fcn.name) not in ['make']:
                    free_functions.append(self.parse_function(fcn))

            namespace_dict['free_functions'] = free_functions

        # sub namespaces
        namespace_dict['namespaces'] = []

        if hasattr(namespace_decl, 'namespaces'):
            sub_namespaces = []
            sub_namespaces_decl = namespace_decl.namespaces(allow_empty=True)
            for ns in sub_namespaces_decl:
                sub_namespaces.append(self.parse_namespace(ns))

            namespace_dict['namespaces'] = sub_namespaces

        return namespace_dict

    def get_header_info(self, namespace_to_parse):
        """
        PyGCCXML header code parser
        magic happens here!
        : returns the parsed header data in python dict
        : return dict keys: namespace, class, io_signature, make,
                       properties, methods
        : Can be used as an CLI command or an extenal API
        """
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
        # try:
        main_namespace = global_namespace
        for ns in namespace_to_parse:
            main_namespace = main_namespace.namespace(ns)
        if main_namespace is None:
            raise BlockToolException('namespace cannot be none')
        self.parsed_data['target_namespace'] = namespace_to_parse

        self.parsed_data['namespace'] = self.parse_namespace(main_namespace)

        # except RuntimeError:
        #     raise BlockToolException(
        #         'Invalid namespace format in the block header file')

        # namespace

        return self.parsed_data

    def run_blocktool(self):
        """ Run, run, run. """
        pass
        # self.get_header_info()