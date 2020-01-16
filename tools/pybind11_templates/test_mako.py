import json
import pathlib
from os import path
from datetime import datetime

current_path = path.dirname(pathlib.Path(__file__).absolute())

filename = 'symbol_sync_cc.json'

with open(path.join(current_path,filename)) as json_file:
    header_info = json.load(json_file)

from pybind_templates import Templates as T

from mako.template import Template

tpl = Template(filename=path.join(current_path,'license.mako'))
license = tpl.render(year=datetime.now().year)

tpl = Template(filename=path.join(current_path,'block_python_hpp.mako'))
print(tpl.render(
    license=license,
    header_info=header_info
))