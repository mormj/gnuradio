basic_sources += join_paths(meson.current_source_dir(), 'vector_source.cpp')

tgt = custom_target('copy_vector_source_header',
                        input : ['vector_source.hpp'],
                        output : ['dummy_output.hpp'], 
                        command : ['python3', join_paths(SCRIPTS_DIR, 'copy_block_header.py'),
                            '--header-file', '@INPUT@', 
                            '--include-path', 'basic/include/gnuradio4/basic', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)

basic_deps += declare_dependency(sources : [tgt[0]] ) 

install_headers('vector_source.hpp', subdir: 'gnuradio4/basic')