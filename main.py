# Copyright 2009-2016 Free Software Foundation, Inc.
# This file is part of GNU Radio
#
# GNU Radio Companion is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# GNU Radio Companion is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA

import argparse, sys

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Gtk

from gnuradio import gr
from .gui.Platform import Platform
from .gui.ActionHandler import ActionHandler


VERSION_AND_DISCLAIMER_TEMPLATE = """\
GNU Radio Companion %s

This program is part of GNU Radio
GRC comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it.
"""


def main():
    parser = argparse.ArgumentParser(
        description=VERSION_AND_DISCLAIMER_TEMPLATE % gr.version())
    parser.add_argument('flow_graphs', nargs='*')
    args = parser.parse_args()

    try:
        Gtk.window_set_default_icon(Gtk.IconTheme().load_icon('gnuradio-grc', 256, 0))
    except:
        pass

    platform = Platform(
        version=gr.version(),
        version_parts=(gr.major_version(), gr.api_version(), gr.minor_version()),
        prefs=gr.prefs(),
        install_prefix=gr.prefix()
    )
    
    app = ActionHandler(args.flow_graphs, platform)
    sys.exit(app.run())
