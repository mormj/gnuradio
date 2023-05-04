# SPDX-FileCopyrightText: 2020 Josh Morman 
#

project('gnuradio', 'cpp', 
  version : '4.0.0.0-preview0',
  license : 'LGPLv3',
  default_options : ['warning_level=3', 'cpp_std=gnu++20'])

## Compilers
cc = meson.get_compiler('cpp') # minimum version 12.1, 12.2 ?
rt_dep = cc.find_library('rt', required : false)
libdl_dep = cc.find_library('dl')
c_available = add_languages('c', required : true)

## Build Dependencies
volk_dep = dependency('volk', version : '>=2.2')
yaml_dep = dependency('yaml-cpp', version : '>=0.6')
fmt_dep = dependency('fmt', version:'8.1.1')
ut_dep = dependency('boost.ut')
spdlog_dep = dependency('spdlog')
threads_dep = dependency('threads')

cmake = import('cmake')
libreflcpp = cmake.subproject('refl-cpp')
reflcpp_dep = libreflcpp.dependency('refl-cpp')

## All python stuff taken out for now

## Subprojects
libpmtv = subproject('pmt')
pmt_dep = libpmtv.get_variable('pmt_dep')
#py3_install_dir_dep = libpmtv.get_variable('py3_install_dir_dep')
CLI11_dep = dependency('cli11')

SCRIPTS_DIR=join_paths(meson.project_source_root(),'utils','blockbuilder','scripts')

## Test Environment
# taken out for now

prefix = get_option('prefix')

GR_DATA_DIR = 'share'
GR_PKG_DATA_DIR = join_paths(GR_DATA_DIR,meson.project_name())

# Set location of config/prefs files in /etc
# Special exception if prefix is /usr so we don't make a /usr/etc.
GR_CONF_DIR = 'etc'
isusr = prefix == '/usr'
if isusr
  SYSCONFDIR = join_paths('/',GR_CONF_DIR)
else
  SYSCONFDIR = join_paths(prefix, GR_CONF_DIR)
endif
GR_PKG_CONF_DIR = join_paths(SYSCONFDIR, meson.project_name(), 'conf.d')
GR_PREFSDIR = join_paths(SYSCONFDIR, meson.project_name(), 'conf.d')
GRC_BLOCKS_DIR = join_paths(GR_PKG_DATA_DIR, 'grc/blocks')

## Subdirectories with meson.build
subdir('gr')
# subdir('kernel')
subdir('blocklib')
subdir('schedulers')
# subdir('utils')
# if (get_option('enable_python'))
#   subdir('python/gnuradio')
#   subdir('grc')
#   subdir('domains/grc')
# endif
if (get_option('enable_testing'))
    subdir('test')
endif
if (get_option('enable_bench'))
    subdir('bench')
endif

summary({'bindir': get_option('bindir'),
        'libdir': get_option('libdir'),
        'datadir': get_option('datadir'),
        'prefix': get_option('prefix'),
        }, section: 'Directories')
