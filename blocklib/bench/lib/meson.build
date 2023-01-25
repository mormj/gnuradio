bench_sources += []
bench_deps += [gnuradio_gr_dep, volk_dep, fmt_dep, spdlog_dep, pmt_dep, python3_embed_dep, json_dep]

block_cpp_args = ['-DHAVE_CPU']

incdir = include_directories(['../include/gnuradio/bench','../include'])
gnuradio_blocklib_bench_lib = library('gnuradio-blocklib-bench', 
    bench_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : bench_deps,
    cpp_args : block_cpp_args)

gnuradio_blocklib_bench_dep = declare_dependency(include_directories : incdir,
					   link_with : gnuradio_blocklib_bench_lib,
                       dependencies : bench_deps)

pkg = import('pkgconfig')
libs = [gnuradio_blocklib_bench_lib]     # the library/libraries users need to link against
h = ['.'] # subdirectories of ${prefix}/${includedir} to add to header path
pkg.generate(libraries : libs,
             subdirs : h,
             version : meson.project_version(),
             name : 'libgnuradio-blocklib-bench',
             filebase : 'gnuradio-blocklib-bench',
             description : 'GNU Radio bench component')
