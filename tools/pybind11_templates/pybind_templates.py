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

from datetime import datetime

''' Templates for generating Python Bindings '''

Templates = {}

# Default licence
Templates['defaultlicense'] = '''
Copyright %d {copyrightholder}.

This is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this software; see the file COPYING.  If not, write to
the Free Software Foundation, Inc., 51 Franklin Street,
Boston, MA 02110-1301, USA.
''' % datetime.now().year

Templates['grlicense'] = '''
Copyright {0} Free Software Foundation, Inc.

This file is part of GNU Radio

GNU Radio is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

GNU Radio is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Radio; see the file COPYING.  If not, write to
the Free Software Foundation, Inc., 51 Franklin Street,
Boston, MA 02110-1301, USA.
'''.format(datetime.now().year)


# Header file of a sync/decimator/interpolator block
Templates['block_python_hpp'] = '''/* -*- c++ -*- */
${str_to_fancyc_comment(license)}
#ifndef INCLUDED_${modname.upper()}_${blockname.upper()}_PYTHON_HPP
#define INCLUDED_${modname.upper()}_${blockname.upper()}_PYTHON_HPP

#include <gnuradio/${grblocktype}.h>
#include <${include_dir_prefix}/${blockname}.h>

void bind_${blockname}(py::module& m)
{
    using ${blockname}    = gr::${modname}::${blockname};

    py::class_<${blockname}, gr::${grblocktype}, std::shared_ptr<${blockname}>>(m, "${blockname}")
        .def(py::init(&${blockname}::make))
% for method in header_info:
      pass
% endfor
        .def("to_basic_block",&${blockname}::to_basic_block)
        ;
} 

#endif /* INCLUDED_${modname.upper()}_${blockname.upper()}_PYTHON_HPP */
'''