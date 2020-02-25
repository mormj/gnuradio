from .base import BindTool
from gnuradio.blocktool import BlockHeaderParser, GenericHeaderParser

import os
import pathlib
import json
from mako.template import Template
from datetime import datetime


class BindingGenerator:

    def __init__(self, **kwargs):
        self.header_extensions = ['.h', '.hh', '.hpp']
        pass

    def get_nonblock_python_hpp(self, header_info, base_name, namespace, prefix_include_root):
        current_path = os.path.dirname(pathlib.Path(__file__).absolute())
        tpl = Template(filename=os.path.join(
            current_path, '..', 'templates', 'license.mako'))
        license = tpl.render(year=datetime.now().year)

        tpl = Template(filename=os.path.join(current_path, '..',
                                             'templates', 'generic_python_hpp.mako'))
        return tpl.render(
            license=license,
            header_info=header_info,
            basename=base_name,
            prefix_include_root=prefix_include_root,
            module=True
        )

    def get_nonblock_python_h(self, header_info, base_name, namespace, prefix_include_root):
        current_path = os.path.dirname(pathlib.Path(__file__).absolute())
        tpl = Template(filename=os.path.join(
            current_path, '..', 'templates', 'license.mako'))
        license = tpl.render(year=datetime.now().year)

        tpl = Template(filename=os.path.join(current_path, '..',
                                             'templates', 'generic_python_h.mako'))
        return tpl.render(
            license=license,
            header_info=header_info,
            basename=base_name,
            prefix_include_root=prefix_include_root,
            module=True
        )

    def get_nonblock_python_cc(self, header_info, base_name, namespace, prefix_include_root):
        current_path = os.path.dirname(pathlib.Path(__file__).absolute())
        tpl = Template(filename=os.path.join(
            current_path, '..', 'templates', 'license.mako'))
        license = tpl.render(year=datetime.now().year)

        tpl = Template(filename=os.path.join(current_path, '..',
                                             'templates', 'generic_python_cc.mako'))
        return tpl.render(
            license=license,
            header_info=header_info,
            basename=base_name,
            prefix_include_root=prefix_include_root,
            module=True
        )

    def bind_from_json(self, pathname):
        base_name = os.path.splitext(os.path.basename(pathname))[0]
        module_dirname = os.path.split(os.path.abspath(os.path.join(os.path.dirname(pathname),'..')))[-1]
        # binding_pathname = '{}_python.hpp'.format(os.path.splitext(pathname)[0])
        # binding_pathname_h = '{}_python.h'.format(os.path.splitext(pathname)[0])
        binding_pathname_cc = '{}_python.cc'.format(os.path.splitext(pathname)[0])
        # Make some assumptions about the namespace
        namespace = ['gr',module_dirname]
        prefix_include_root = 'gnuradio'

        with open(pathname,'r') as fp:
            header_info = json.load(fp)
            # pybind_code = self.get_nonblock_python_hpp(
            #     header_info, base_name, namespace, prefix_include_root)
            # with open(binding_pathname, 'w+') as outfile:
            #     outfile.write(pybind_code)

            # pybind_code_h = self.get_nonblock_python_h(
            #     header_info, base_name, namespace, prefix_include_root)
            # with open(binding_pathname, 'w+') as outfile:
            #     outfile.write(pybind_code_h)

            pybind_code_cc = self.get_nonblock_python_cc(
                header_info, base_name, namespace, prefix_include_root)
            with open(binding_pathname_cc, 'w+') as outfile:
                outfile.write(pybind_code_cc)
            return binding_pathname_cc
            
    def write_bindings_generic(self, module_path, base_name, header_info, output_dir, namespace, prefix_include_root):
        json_pathname = os.path.join(output_dir, '{}.json'.format(base_name))
        with open(json_pathname, 'w') as outfile:
            json.dump(header_info, outfile)

        # binding_pathname = os.path.join(
        #     output_dir, '{}_python.hpp'.format(base_name))
        # binding_pathname_h = os.path.join(
        #     output_dir, '{}_python.h'.format(base_name))
        binding_pathname_cc = os.path.join(
            output_dir, '{}_python.cc'.format(base_name))


        try:
            # pybind_code = self.get_nonblock_python_hpp(
            #     header_info, base_name, namespace, prefix_include_root)
            # with open(binding_pathname, 'w+') as outfile:
            #     outfile.write(pybind_code)
            # pybind_code = self.get_nonblock_python_h(
            #     header_info, base_name, namespace, prefix_include_root)
            # with open(binding_pathname_h, 'w+') as outfile:
            #     outfile.write(pybind_code)
            pybind_code = self.get_nonblock_python_cc(
                header_info, base_name, namespace, prefix_include_root)
            with open(binding_pathname_cc, 'w+') as outfile:
                outfile.write(pybind_code)
            return binding_pathname
        except:
            return None

    def process_generic_header_file(self, file_to_process, module_path, prefix, output_dir, namespace, prefix_include_root):
        binding_pathname = None
        # module_include_path = os.path.abspath(os.path.join(module_path, 'include'))
        base_name = os.path.splitext(os.path.basename(file_to_process))[0]
        module_include_path = os.path.abspath(os.path.dirname(module_path))
        top_include_path = os.path.join(
            module_include_path.split('include'+os.path.sep)[0], 'include')
        # blocks_include_path=os.path.abspath(os.path.join(module_path,'..','gr-blocks','include'))
        # gr_include_path=os.path.abspath(os.path.join(module_path,'..','gnuradio-runtime','include'))
        # include_paths = ','.join((module_include_path,blocks_include_path,gr_include_path))
        prefix_include_path = os.path.abspath(os.path.join(prefix, 'include'))
        include_paths = ','.join(
            (prefix_include_path, module_include_path, top_include_path))
        parser = GenericHeaderParser(
            include_paths=include_paths, file_path=file_to_process)
        try:
            header_info = parser.get_header_info(namespace)
            binding_pathname = self.write_bindings_generic(
                module_path, base_name, header_info, output_dir, namespace, prefix_include_root)
        except Exception as e:
            print(e)
            failure_pathname = os.path.join(
                output_dir, 'failed_conversions.txt')
            with open(failure_pathname, 'a+') as outfile:
                outfile.write(file_to_process)
                outfile.write(str(e))
                outfile.write('\n')

        return binding_pathname

    def process_file(self, pathname, prefix, output_dir, namespace, prefix_include_root):
        # path that the file lives in
        module_path = pathlib.Path(pathname).absolute()

        # output_dir = os.path.join(args.output,  os.path.split(os.path.dirname(file)[1]))
        if 'include'+os.path.sep in pathname:
            rel_path_after_include = os.path.split(
                pathname.split('include'+os.path.sep)[-1])[0]
            output_dir = os.path.join(output_dir, rel_path_after_include, 'bindings')
            # output_dir = os.path.join(output_dir,os.path.basename(file))
            if output_dir and not os.path.exists(output_dir):
                output_dir = os.path.abspath(output_dir)
                print('creating directory {}'.format(output_dir))
                os.makedirs(output_dir)

            return self.process_generic_header_file(pathname,
                                                    module_path, prefix, output_dir, namespace, prefix_include_root)

    def gen_top_level_cpp(self, file_list, output_dir, module_name):
        current_path = os.path.dirname(pathlib.Path(__file__).absolute())
        file = file_list[0]
        if 'include'+os.path.sep in file:
            rel_path_after_include = os.path.split(
                file.split('include'+os.path.sep)[-1])[0]
            output_dir = os.path.join(output_dir, rel_path_after_include,'bindings')
            if output_dir and not os.path.exists(output_dir):
                output_dir = os.path.abspath(output_dir)
                print('creating directory {}'.format(output_dir))
                os.makedirs(output_dir)

        tpl = Template(filename=os.path.join(
            current_path, '..', 'templates', 'license.mako'))
        license = tpl.render(year=datetime.now().year)

        binding_pathname = os.path.join(output_dir, 'python_bindings.cc')
        file_list = [os.path.split(f)[-1] for f in file_list]
        tpl = Template(filename=os.path.join(current_path, '..',
                                             'templates', 'python_bindings_cc.mako'))
        pybind_code = tpl.render(
            license=license,
            files=file_list,
            module_name=module_name
        )

        # print(pybind_code)
        try:
            with open(binding_pathname, 'w+') as outfile:
                outfile.write(pybind_code)
            return binding_pathname
        except:
            return None

    def gen_cmake_lists(self, file_list, output_dir, module_name):
        current_path = os.path.dirname(pathlib.Path(__file__).absolute())
        file = file_list[0]
        if 'include'+os.path.sep in file:
            rel_path_after_include = os.path.split(
                file.split('include'+os.path.sep)[-1])[0]
            output_dir = os.path.join(output_dir, rel_path_after_include,'bindings')
            if output_dir and not os.path.exists(output_dir):
                output_dir = os.path.abspath(output_dir)
                print('creating directory {}'.format(output_dir))
                os.makedirs(output_dir)

        tpl = Template(filename=os.path.join(
            current_path, '..', 'templates', 'license.mako'))
        license = tpl.render(year=datetime.now().year)

        binding_pathname = os.path.join(output_dir, 'CMakeLists.txt')
        file_list = [os.path.split(f)[-1] for f in file_list]
        tpl = Template(filename=os.path.join(current_path, '..',
                                             'templates', 'CMakeLists.txt.mako'))
        pybind_code = tpl.render(
            license=license,
            files=file_list,
            module_name=module_name
        )

        # print(pybind_code)
        try:
            with open(binding_pathname, 'w+') as outfile:
                outfile.write(pybind_code)
            return binding_pathname
        except:
            return None

    def get_file_list(self, include_path):
        file_list = []
        for root, _, files in os.walk(include_path):
            for file in files:
                _, file_extension = os.path.splitext(file)
                if (file_extension in self.header_extensions):
                    pathname = os.path.abspath(os.path.join(root, file))
                    file_list.append(pathname)
        return file_list

    def gen_bindings(self, module_dir, prefix, namespace, prefix_include_root, output_dir):
        file_list = sorted(self.get_file_list(module_dir))
        api_pathnames = [s for s in file_list if 'api.h' in s]
        for f in api_pathnames:
            file_list.remove(f)
        self.gen_top_level_cpp(file_list, output_dir, namespace[-1])
        self.gen_cmake_lists(file_list, output_dir, namespace[-1])
        for fn in file_list:
            self.process_file(fn, prefix, output_dir,
                              namespace, prefix_include_root)