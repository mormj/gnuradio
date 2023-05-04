basic_sources += join_paths(meson.current_source_dir(), 'multiply_const.cpp')

tgt = custom_target('copy_multiply_const_header',
                        input : ['multiply_const.hpp'],
                        output : ['dummy_output.hpp'], 
                        command : ['python3', join_paths(SCRIPTS_DIR, 'copy_block_header.py'),
                            '--header-file', '@INPUT@', 
                            '--include-path', 'basic/include/gnuradio4/basic', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)

basic_deps += declare_dependency(sources : [tgt[0]] ) 

install_headers('multiply_const.hpp', subdir: 'gnuradio4/basic')

if (GR_ENABLE_PYTHON)
    basic_pybind_sources += files('multiply_const_pybind.cpp')
endif