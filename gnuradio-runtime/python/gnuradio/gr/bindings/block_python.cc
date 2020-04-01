/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/* This file is automatically generated using bindtool */

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/block.h>
// pydoc.h is automatically generated in the build directory
#include <block_pydoc.h>

void bind_block(py::module& m)
{
    using block    = gr::block;



    py::class_<block, gr::basic_block,
        std::shared_ptr<block>>(m, "block", D(block))



        .def("history",&block::history,
            D(block,history)
        )


        .def("set_history",&block::set_history,
            py::arg("history"),
            D(block,set_history)
        )
        .def("declare_sample_delay",(void (block::*)(int, unsigned int))&block::declare_sample_delay,
            py::arg("which"),
            py::arg("delay"),
            D(block,declare_sample_delay,0)
        )
        .def("declare_sample_delay",(void (block::*)(unsigned int))&block::declare_sample_delay,
            py::arg("delay"),
            D(block,declare_sample_delay,1)
        )
        .def("sample_delay",&block::sample_delay,
            py::arg("which"),
            D(block,sample_delay)
        )


        .def("fixed_rate",&block::fixed_rate,
            D(block,fixed_rate)
        )


        .def("forecast",&block::forecast,
            py::arg("noutput_items"),
            py::arg("ninput_items_required"),
            D(block,forecast)
        )
        .def("general_work",&block::general_work,
            py::arg("noutput_items"),
            py::arg("ninput_items"),
            py::arg("input_items"),
            py::arg("output_items"),
            D(block,general_work)
        )


        .def("start",&block::start,
            D(block,start)
        )


        .def("stop",&block::stop,
            D(block,stop)
        )


        .def("set_output_multiple",&block::set_output_multiple,
            py::arg("multiple"),
            D(block,set_output_multiple)
        )


        .def("output_multiple",&block::output_multiple,
            D(block,output_multiple)
        )


        .def("output_multiple_set",&block::output_multiple_set,
            D(block,output_multiple_set)
        )


        .def("set_alignment",&block::set_alignment,
            py::arg("multiple"),
            D(block,set_alignment)
        )


        .def("alignment",&block::alignment,
            D(block,alignment)
        )


        .def("set_unaligned",&block::set_unaligned,
            py::arg("na"),
            D(block,set_unaligned)
        )


        .def("unaligned",&block::unaligned,
            D(block,unaligned)
        )


        .def("set_is_unaligned",&block::set_is_unaligned,
            py::arg("u"),
            D(block,set_is_unaligned)
        )


        .def("is_unaligned",&block::is_unaligned,
            D(block,is_unaligned)
        )


        .def("consume",&block::consume,
            py::arg("which_input"),
            py::arg("how_many_items"),
            D(block,consume)
        )
        .def("consume_each",&block::consume_each,
            py::arg("how_many_items"),
            D(block,consume_each)
        )
        .def("produce",&block::produce,
            py::arg("which_output"),
            py::arg("how_many_items"),
            D(block,produce)
        )
        .def("set_relative_rate",(void (block::*)(double))&block::set_relative_rate,
            py::arg("relative_rate"),
            D(block,set_relative_rate,0)
        )
        .def("set_inverse_relative_rate",&block::set_inverse_relative_rate,
            py::arg("inverse_relative_rate"),
            D(block,set_inverse_relative_rate)
        )
        .def("set_relative_rate",(void (block::*)(uint64_t, uint64_t))&block::set_relative_rate,
            py::arg("interpolation"),
            py::arg("decimation"),
            D(block,set_relative_rate,1)
        )


        .def("relative_rate",&block::relative_rate,
            D(block,relative_rate)
        )


        .def("relative_rate_i",&block::relative_rate_i,
            D(block,relative_rate_i)
        )


        .def("relative_rate_d",&block::relative_rate_d,
            D(block,relative_rate_d)
        )


        .def("mp_relative_rate",&block::mp_relative_rate,
            D(block,mp_relative_rate)
        )


        .def("fixed_rate_ninput_to_noutput",&block::fixed_rate_ninput_to_noutput,
            py::arg("ninput"),
            D(block,fixed_rate_ninput_to_noutput)
        )
        .def("fixed_rate_noutput_to_ninput",&block::fixed_rate_noutput_to_ninput,
            py::arg("noutput"),
            D(block,fixed_rate_noutput_to_ninput)
        )
        .def("nitems_read",&block::nitems_read,
            py::arg("which_input"),
            D(block,nitems_read)
        )
        .def("nitems_written",&block::nitems_written,
            py::arg("which_output"),
            D(block,nitems_written)
        )


        .def("tag_propagation_policy",&block::tag_propagation_policy,
            D(block,tag_propagation_policy)
        )


        .def("set_tag_propagation_policy",&block::set_tag_propagation_policy,
            py::arg("p"),
            D(block,set_tag_propagation_policy)
        )


        .def("min_noutput_items",&block::min_noutput_items,
            D(block,min_noutput_items)
        )


        .def("set_min_noutput_items",&block::set_min_noutput_items,
            py::arg("m"),
            D(block,set_min_noutput_items)
        )


        .def("max_noutput_items",&block::max_noutput_items,
            D(block,max_noutput_items)
        )


        .def("set_max_noutput_items",&block::set_max_noutput_items,
            py::arg("m"),
            D(block,set_max_noutput_items)
        )


        .def("unset_max_noutput_items",&block::unset_max_noutput_items,
            D(block,unset_max_noutput_items)
        )


        .def("is_set_max_noutput_items",&block::is_set_max_noutput_items,
            D(block,is_set_max_noutput_items)
        )


        .def("expand_minmax_buffer",&block::expand_minmax_buffer,
            py::arg("port"),
            D(block,expand_minmax_buffer)
        )
        .def("max_output_buffer",&block::max_output_buffer,
            py::arg("i"),
            D(block,max_output_buffer)
        )
        .def("set_max_output_buffer",(void (block::*)(long int))&block::set_max_output_buffer,
            py::arg("max_output_buffer"),
            D(block,set_max_output_buffer,0)
        )
        .def("set_max_output_buffer",(void (block::*)(int, long int))&block::set_max_output_buffer,
            py::arg("port"),
            py::arg("max_output_buffer"),
            D(block,set_max_output_buffer,1)
        )
        .def("min_output_buffer",&block::min_output_buffer,
            py::arg("i"),
            D(block,min_output_buffer)
        )
        .def("set_min_output_buffer",(void (block::*)(long int))&block::set_min_output_buffer,
            py::arg("min_output_buffer"),
            D(block,set_min_output_buffer,0)
        )
        .def("set_min_output_buffer",(void (block::*)(int, long int))&block::set_min_output_buffer,
            py::arg("port"),
            py::arg("min_output_buffer"),
            D(block,set_min_output_buffer,1)
        )


        .def("pc_noutput_items",&block::pc_noutput_items,
            D(block,pc_noutput_items)
        )


        .def("pc_noutput_items_avg",&block::pc_noutput_items_avg,
            D(block,pc_noutput_items_avg)
        )


        .def("pc_noutput_items_var",&block::pc_noutput_items_var,
            D(block,pc_noutput_items_var)
        )


        .def("pc_nproduced",&block::pc_nproduced,
            D(block,pc_nproduced)
        )


        .def("pc_nproduced_avg",&block::pc_nproduced_avg,
            D(block,pc_nproduced_avg)
        )


        .def("pc_nproduced_var",&block::pc_nproduced_var,
            D(block,pc_nproduced_var)
        )


        .def("pc_input_buffers_full",(float (block::*)(int))&block::pc_input_buffers_full,
            py::arg("which"),
            D(block,pc_input_buffers_full,0)
        )
        .def("pc_input_buffers_full_avg",(float (block::*)(int))&block::pc_input_buffers_full_avg,
            py::arg("which"),
            D(block,pc_input_buffers_full_avg,0)
        )
        .def("pc_input_buffers_full_var",(float (block::*)(int))&block::pc_input_buffers_full_var,
            py::arg("which"),
            D(block,pc_input_buffers_full_var,0)
        )


        .def("pc_input_buffers_full",(std::vector<float, std::allocator<float> > (block::*)())&block::pc_input_buffers_full,
            D(block,pc_input_buffers_full,1)
        )


        .def("pc_input_buffers_full_avg",(std::vector<float, std::allocator<float> > (block::*)())&block::pc_input_buffers_full_avg,
            D(block,pc_input_buffers_full_avg,1)
        )


        .def("pc_input_buffers_full_var",(std::vector<float, std::allocator<float> > (block::*)())&block::pc_input_buffers_full_var,
            D(block,pc_input_buffers_full_var,1)
        )


        .def("pc_output_buffers_full",(float (block::*)(int))&block::pc_output_buffers_full,
            py::arg("which"),
            D(block,pc_output_buffers_full,0)
        )
        .def("pc_output_buffers_full_avg",(float (block::*)(int))&block::pc_output_buffers_full_avg,
            py::arg("which"),
            D(block,pc_output_buffers_full_avg,0)
        )
        .def("pc_output_buffers_full_var",(float (block::*)(int))&block::pc_output_buffers_full_var,
            py::arg("which"),
            D(block,pc_output_buffers_full_var,0)
        )


        .def("pc_output_buffers_full",(std::vector<float, std::allocator<float> > (block::*)())&block::pc_output_buffers_full,
            D(block,pc_output_buffers_full,1)
        )


        .def("pc_output_buffers_full_avg",(std::vector<float, std::allocator<float> > (block::*)())&block::pc_output_buffers_full_avg,
            D(block,pc_output_buffers_full_avg,1)
        )


        .def("pc_output_buffers_full_var",(std::vector<float, std::allocator<float> > (block::*)())&block::pc_output_buffers_full_var,
            D(block,pc_output_buffers_full_var,1)
        )


        .def("pc_work_time",&block::pc_work_time,
            D(block,pc_work_time)
        )


        .def("pc_work_time_avg",&block::pc_work_time_avg,
            D(block,pc_work_time_avg)
        )


        .def("pc_work_time_var",&block::pc_work_time_var,
            D(block,pc_work_time_var)
        )


        .def("pc_work_time_total",&block::pc_work_time_total,
            D(block,pc_work_time_total)
        )


        .def("pc_throughput_avg",&block::pc_throughput_avg,
            D(block,pc_throughput_avg)
        )


        .def("reset_perf_counters",&block::reset_perf_counters,
            D(block,reset_perf_counters)
        )


        .def("setup_pc_rpc",&block::setup_pc_rpc,
            D(block,setup_pc_rpc)
        )


        .def("is_pc_rpc_set",&block::is_pc_rpc_set,
            D(block,is_pc_rpc_set)
        )


        .def("no_pc_rpc",&block::no_pc_rpc,
            D(block,no_pc_rpc)
        )


        .def("set_processor_affinity",&block::set_processor_affinity,
            py::arg("mask"),
            D(block,set_processor_affinity)
        )


        .def("unset_processor_affinity",&block::unset_processor_affinity,
            D(block,unset_processor_affinity)
        )


        .def("processor_affinity",&block::processor_affinity,
            D(block,processor_affinity)
        )


        .def("active_thread_priority",&block::active_thread_priority,
            D(block,active_thread_priority)
        )


        .def("thread_priority",&block::thread_priority,
            D(block,thread_priority)
        )


        .def("set_thread_priority",&block::set_thread_priority,
            py::arg("priority"),
            D(block,set_thread_priority)
        )


        .def("update_rate",&block::update_rate,
            D(block,update_rate)
        )


        .def("system_handler",&block::system_handler,
            py::arg("msg"),
            D(block,system_handler)
        )
        .def("set_log_level",&block::set_log_level,
            py::arg("level"),
            D(block,set_log_level)
        )


        .def("log_level",&block::log_level,
            D(block,log_level)
        )


        .def("finished",&block::finished,
            D(block,finished)
        )


        .def("detail",&block::detail,
            D(block,detail)
        )


        .def("set_detail",&block::set_detail,
            py::arg("detail"),
            D(block,set_detail)
        )


        .def("notify_msg_neighbors",&block::notify_msg_neighbors,
            D(block,notify_msg_neighbors)
        )


        .def("clear_finished",&block::clear_finished,
            D(block,clear_finished)
        )


        .def("identifier",&block::identifier,
            D(block,identifier)
        )

        ;

    py::enum_<gr::block::work_return_t>(m,"work_return_t")
        .value("WORK_CALLED_PRODUCE", gr::block::WORK_CALLED_PRODUCE) // -2
        .value("WORK_DONE", gr::block::WORK_DONE) // -1
        .export_values()
    ;
    py::enum_<gr::block::tag_propagation_policy_t>(m,"tag_propagation_policy_t")
        .value("TPP_DONT", gr::block::TPP_DONT) // 0
        .value("TPP_ALL_TO_ALL", gr::block::TPP_ALL_TO_ALL) // 1
        .value("TPP_ONE_TO_ONE", gr::block::TPP_ONE_TO_ONE) // 2
        .value("TPP_CUSTOM", gr::block::TPP_CUSTOM) // 3
        .export_values()
    ;


        m.def("cast_to_block_sptr",&::gr::cast_to_block_sptr,
            py::arg("p"),
            D(cast_to_block_sptr)
        );


        py::module m_messages = m.def_submodule("messages");






        py::module m_thread = m.def_submodule("thread");







}







