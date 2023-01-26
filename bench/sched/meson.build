if CLI11_dep.found()
srcs = ['bm_a.cpp']
executable('bm_a',
    srcs,
    # include_directories : incdir,
    link_language : 'cpp',
    dependencies: [gnuradio_gr_dep,
                   gnuradio_blocklib_blocks_dep,
                   gnuradio_blocklib_bench_dep,
                   gnuradio_blocklib_streamops_dep,
                   gnuradio_scheduler_nbt_dep,
                   CLI11_dep],
    install : true)
srcs = ['bm_b.cpp']
executable('bm_b',
    srcs,
    # include_directories : incdir,
    link_language : 'cpp',
    dependencies: [gnuradio_gr_dep,
                   gnuradio_blocklib_blocks_dep,
                   gnuradio_blocklib_math_dep,
                   gnuradio_blocklib_bench_dep,
                   gnuradio_blocklib_streamops_dep,
                   gnuradio_scheduler_nbt_dep,
                   CLI11_dep],
    install : true)
endif