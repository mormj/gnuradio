
graph_dep = declare_dependency(include_directories: '..', dependencies : [vir_dep, fmt_dep, reflcpp_dep, pmt_dep])

header_files = [
    'buffer_skeleton.hpp',
    'buffer.hpp',
    'circular_buffer.hpp',
    'claim_strategy.hpp',
    'graph.hpp',
    'node_traits.hpp',
    'node.hpp',
    'port_traits.hpp',
    'port.hpp',
    'sequence.hpp',
    'typelist.hpp',
    'utils.hpp',
    'wait_strategy.hpp'
]
install_headers(header_files, subdir : 'gnuradio4', preserve_path: true)