basic_sources = []
basic_deps = []
basic_pybind_sources = []

subdir('multiply_const')
subdir('vector_source')
subdir('vector_sink')
subdir('include/gnuradio4/basic')
subdir('lib')

if GR_ENABLE_PYTHON
    subdir('python/gnuradio4/basic')
endif

if (get_option('enable_testing'))
    subdir('test')
endif