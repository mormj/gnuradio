test_names = [
    'test_basic'
]

deps = [graph_dep, fmt_dep, ut_dep, reflcpp_dep, 
        gnuradio_blocklib_basic_dep]
cpp_args = '-march=native'
foreach test_name : test_names
test_exe = executable(test_name, test_name + '.cpp',
  cpp_args : cpp_args,
  link_args : cpp_args,
  dependencies : [deps],
  install: false)
endforeach