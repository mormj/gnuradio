basic_deps += [graph_dep]

incdir = include_directories(['../include'])
gnuradio_blocklib_basic_lib = library('gnuradio-blocklib-blocks', 
    basic_sources, 
    include_directories: [incdir],
    install : true,
    link_language: 'cpp',
    dependencies : basic_deps)

gnuradio_blocklib_basic_dep = declare_dependency(
					   link_with : gnuradio_blocklib_basic_lib,
             include_directories: [incdir],
                       dependencies : basic_deps)

