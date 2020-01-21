

/* Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */


/* This file is automatically generated using the gen_nonblock_bindings.py tool */

#ifndef INCLUDED_GR_BLOCK_PYTHON_HPP
#define INCLUDED_GR_BLOCK_PYTHON_HPP

#include <gnuradio/block.h>

void bind_block(py::module& m)
{
    using block    = gr::block;

    py::class_<block, std::shared_ptr<block>>(m, "block")
        .def("history",&block::history)
        .def("set_history",&block::set_history,
            py::arg("history") 
        )
        .def("declare_sample_delay",(void (block::*)(int, unsigned))&block::declare_sample_delay,
            py::arg("which"), 
            py::arg("delay") 
        )
        .def("declare_sample_delay",(void (block::*)(unsigned))&block::declare_sample_delay,
            py::arg("delay") 
        )
        .def("sample_delay",&block::sample_delay,
            py::arg("which") 
        )
        .def("fixed_rate",&block::fixed_rate)
        .def("forecast",&block::forecast,
            py::arg("noutput_items"), 
            py::arg("ninput_items_required") 
        )
        .def("general_work",&block::general_work,
            py::arg("noutput_items"), 
            py::arg("ninput_items"), 
            py::arg("input_items"), 
            py::arg("output_items") 
        )
        .def("start",&block::start)
        .def("stop",&block::stop)
        .def("set_output_multiple",&block::set_output_multiple,
            py::arg("multiple") 
        )
        .def("output_multiple",&block::output_multiple)
        .def("output_multiple_set",&block::output_multiple_set)
        .def("set_alignment",&block::set_alignment,
            py::arg("multiple") 
        )
        .def("alignment",&block::alignment)
        .def("set_unaligned",&block::set_unaligned,
            py::arg("na") 
        )
        .def("unaligned",&block::unaligned)
        .def("set_is_unaligned",&block::set_is_unaligned,
            py::arg("u") 
        )
        .def("is_unaligned",&block::is_unaligned)
        .def("consume",&block::consume,
            py::arg("which_input"), 
            py::arg("how_many_items") 
        )
        .def("consume_each",&block::consume_each,
            py::arg("how_many_items") 
        )
        .def("produce",&block::produce,
            py::arg("which_output"), 
            py::arg("how_many_items") 
        )
        .def("set_relative_rate",(void (block::*)(double))&block::set_relative_rate,
            py::arg("relative_rate") 
        )
        .def("set_inverse_relative_rate",&block::set_inverse_relative_rate,
            py::arg("inverse_relative_rate") 
        )
        .def("set_relative_rate",(void (block::*)(uint64_t, uint64_t))&block::set_relative_rate,
            py::arg("interpolation"), 
            py::arg("decimation") 
        )
        .def("relative_rate",&block::relative_rate)
        .def("relative_rate_i",&block::relative_rate_i)
        .def("relative_rate_d",&block::relative_rate_d)
        .def("mp_relative_rate",&block::mp_relative_rate)
        .def("fixed_rate_ninput_to_noutput",&block::fixed_rate_ninput_to_noutput,
            py::arg("ninput") 
        )
        .def("fixed_rate_noutput_to_ninput",&block::fixed_rate_noutput_to_ninput,
            py::arg("noutput") 
        )
        .def("nitems_read",&block::nitems_read,
            py::arg("which_input") 
        )
        .def("nitems_written",&block::nitems_written,
            py::arg("which_output") 
        )
        .def("tag_propagation_policy",&block::tag_propagation_policy)
        .def("set_tag_propagation_policy",&block::set_tag_propagation_policy,
            py::arg("p") 
        )
        .def("min_noutput_items",&block::min_noutput_items)
        .def("set_min_noutput_items",&block::set_min_noutput_items,
            py::arg("m") 
        )
        .def("max_noutput_items",&block::max_noutput_items)
        .def("set_max_noutput_items",&block::set_max_noutput_items,
            py::arg("m") 
        )
        .def("unset_max_noutput_items",&block::unset_max_noutput_items)
        .def("is_set_max_noutput_items",&block::is_set_max_noutput_items)
        .def("expand_minmax_buffer",&block::expand_minmax_buffer,
            py::arg("port") 
        )
        .def("max_output_buffer",&block::max_output_buffer,
            py::arg("i") 
        )
        .def("set_max_output_buffer",(void (block::*)(long))&block::set_max_output_buffer,
            py::arg("max_output_buffer") 
        )
        .def("set_max_output_buffer",(void (block::*)(int, long))&block::set_max_output_buffer,
            py::arg("port"), 
            py::arg("max_output_buffer") 
        )
        .def("min_output_buffer",&block::min_output_buffer,
            py::arg("i") 
        )
        .def("set_min_output_buffer",(void (block::*)(long))&block::set_min_output_buffer,
            py::arg("min_output_buffer") 
        )
        .def("set_min_output_buffer",(void (block::*)(int, long))&block::set_min_output_buffer,
            py::arg("port"), 
            py::arg("min_output_buffer") 
        )
        .def("pc_noutput_items",&block::pc_noutput_items)
        .def("pc_noutput_items_avg",&block::pc_noutput_items_avg)
        .def("pc_noutput_items_var",&block::pc_noutput_items_var)
        .def("pc_nproduced",&block::pc_nproduced)
        .def("pc_nproduced_avg",&block::pc_nproduced_avg)
        .def("pc_nproduced_var",&block::pc_nproduced_var)
        .def("pc_input_buffers_full",(float (block::*)(int))&block::pc_input_buffers_full,
            py::arg("which") 
        )
        .def("pc_input_buffers_full_avg",(float (block::*)(int))&block::pc_input_buffers_full_avg,
            py::arg("which") 
        )
        .def("pc_input_buffers_full_var",(float (block::*)(int))&block::pc_input_buffers_full_var,
            py::arg("which") 
        )
        .def("pc_input_buffers_full",(std::vector<float> (block::*)())&block::pc_input_buffers_full)
        .def("pc_input_buffers_full_avg",(std::vector<float> (block::*)())&block::pc_input_buffers_full_avg)
        .def("pc_input_buffers_full_var",(std::vector<float> (block::*)())&block::pc_input_buffers_full_var)
        .def("pc_output_buffers_full",(float (block::*)(int))&block::pc_output_buffers_full,
            py::arg("which") 
        )
        .def("pc_output_buffers_full_avg",(float (block::*)(int))&block::pc_output_buffers_full_avg,
            py::arg("which") 
        )
        .def("pc_output_buffers_full_var",(float (block::*)(int))&block::pc_output_buffers_full_var,
            py::arg("which") 
        )
        .def("pc_output_buffers_full",(std::vector<float> (block::*)())&block::pc_output_buffers_full)
        .def("pc_output_buffers_full_avg",(std::vector<float> (block::*)())&block::pc_output_buffers_full_avg)
        .def("pc_output_buffers_full_var",(std::vector<float> (block::*)())&block::pc_output_buffers_full_var)
        .def("pc_work_time",&block::pc_work_time)
        .def("pc_work_time_avg",&block::pc_work_time_avg)
        .def("pc_work_time_var",&block::pc_work_time_var)
        .def("pc_work_time_total",&block::pc_work_time_total)
        .def("pc_throughput_avg",&block::pc_throughput_avg)
        .def("reset_perf_counters",&block::reset_perf_counters)
        .def("setup_pc_rpc",&block::setup_pc_rpc)
        .def("is_pc_rpc_set",&block::is_pc_rpc_set)
        .def("no_pc_rpc",&block::no_pc_rpc)
        .def("set_processor_affinity",&block::set_processor_affinity,
            py::arg("mask") 
        )
        .def("unset_processor_affinity",&block::unset_processor_affinity)
        .def("processor_affinity",&block::processor_affinity)
        .def("active_thread_priority",&block::active_thread_priority)
        .def("thread_priority",&block::thread_priority)
        .def("set_thread_priority",&block::set_thread_priority,
            py::arg("priority") 
        )
        .def("update_rate",&block::update_rate)
        .def("system_handler",&block::system_handler,
            py::arg("msg") 
        )
        .def("set_log_level",&block::set_log_level,
            py::arg("level") 
        )
        .def("log_level",&block::log_level)
        .def("finished",&block::finished)
        .def("detail",&block::detail)
        .def("set_detail",&block::set_detail,
            py::arg("detail") 
        )
        .def("notify_msg_neighbors",&block::notify_msg_neighbors)
        .def("clear_finished",&block::clear_finished)
        .def("identifier",&block::identifier)
        ;

    m.def("cast_to_block_sptr",&gr::cast_to_block_sptr,
        py::arg("p") 
    );
} 

#endif /* INCLUDED_GR_BLOCK_PYTHON_HPP */
