import json
import pathlib
from os import path
from datetime import datetime
import os

current_path = path.dirname(pathlib.Path(__file__).absolute())

prefix_include_root = 'gnuradio'

# filename = 'symbol_sync_cc.json'
# filename = '/share/tmp/blocktool_pybind/file_source.h/file_source.json'
# filename = '/share/tmp/blocktool_pybind/pmt.h/pmt.json'; base_name = 'pmt'
# filename = '/share/tmp/blocktool_pybind/pmt_pool.h/pmt_pool.json'; base_name = 'pmt_pool'
# filename = '/share/tmp/blocktool_pybind/gnuradio/logger.json'; base_name = 'logger'
filename = '/share/tmp/blocktool_pybind/gnuradio/blocks/file_meta_sink.json'; base_name = 'file_meta_sink'
# base_name = 'pmt_pool'
#with open(path.join(current_path,filename)) as json_file:
with open(filename) as json_file:
    header_info = json.load(json_file)

from pybind_templates import Templates as T

from mako.template import Template

tpl = Template(filename=path.join(current_path,'license.mako'))
license = tpl.render(year=datetime.now().year)

tpl = Template(filename=os.path.join(current_path,'nonblock_python_hpp.mako'))
print(tpl.render(
    license=license,
    header_info=header_info,
    basename = base_name,
    namespace = ['pmt'],
    prefix_include_root = prefix_include_root
))