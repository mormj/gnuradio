######################
#  Python Bindings ###
######################

srcs = ['__init__.py']

# This is for testing without installing
foreach s: srcs
configure_file(copy: true,
    input: s,
    output: s
)
endforeach

basic_pybind_sources += 'basic_pybind.cpp'

gnuradio_blocklib_basic_pybind = py3.extension_module('basic_python',
    basic_pybind_sources,
    include_directories: ['../../../lib', incdir_numpy],
    dependencies : [graph_dep, gnuradio_blocklib_basic_dep, python3_dep, pybind11_dep],
    link_language : 'cpp',
    install : true,
    install_dir : join_paths(py3_install_dir,'gnuradio4','basic')
)

gnuradio_blocklib_basic_pybind_dep = declare_dependency(include_directories : incdir,
					   link_with : gnuradio_blocklib_basic_pybind,
                       dependencies : basic_deps)


# Target for pure python
py3.install_sources(srcs, subdir : join_paths('gnuradio4','basic'), pure : py3_purelib)

