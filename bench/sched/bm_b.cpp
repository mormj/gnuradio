#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/bench/copy.h>
#include <gnuradio/blocks/null_sink.h>
#include <gnuradio/blocks/null_source.h>
#include <gnuradio/flowgraph.h>
#include <gnuradio/math/multiply_const.h>
#include <gnuradio/realtime.h>
#include <gnuradio/runtime.h>
#include <gnuradio/schedulers/nbt/scheduler_nbt.h>
#include <gnuradio/streamops/head.h>

#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

using namespace gr;

int main(int argc, char* argv[])
{
    uint64_t samples = 1000000000;
    unsigned int stages = 1;
    unsigned int nthreads = 0;
    bool rt_prio = false;
    size_t buffer_size = 32768;

    std::vector<unsigned int> cpu_affinity;

    CLI::App app{ "App description" };

    // app.add_option("-h,--help", "display help");
    app.add_option("--samples", samples, "Number of Samples");
    app.add_option("--stages", stages, "Number of copy blocks");
    app.add_option("--buffer_size", buffer_size, "Default Buffer Size");
    app.add_option("--nthreads", nthreads, "Number of threads (0: tpb)");
    app.add_flag("--rt_prio", rt_prio, "Enable Real-time priority");

    CLI11_PARSE(app, argc, argv);

    if (rt_prio && gr::enable_realtime_scheduling() != rt_status_t::OK) {
        std::cout << "Error: failed to enable real-time scheduling." << std::endl;
    }

    {
        auto src = blocks::null_source::make({ 1, sizeof(float) });
        auto head = streamops::head::make_cpu({ samples, sizeof(float) });
        auto snk = blocks::null_sink::make({ 1, sizeof(float) });
        gr::node_sptr last_node = src;

        std::vector<block_sptr> block_group;
        block_group.push_back(src);

        auto fg = flowgraph::make();
        for (unsigned int i = 0; i < stages; i++) {
            // auto b1 = math::multiply_const_ff::make({ 1.0 });
            auto b2 = math::multiply_const_ff::make({ 2.0 });
            auto b3 = math::multiply_const_ff::make({ 0.5 });
            auto b4 = math::multiply_const_ff::make({ -1.0 });

            // fg->connect(last_node, 0, b1, 0);
            fg->connect(last_node, 0, b2, 0);
            fg->connect(b2, 0, b3, 0);
            fg->connect(b3, 0, b4, 0);

            last_node = b4;

            // block_group.push_back(b1);
            block_group.push_back(b2);
            block_group.push_back(b3);
            block_group.push_back(b4);
        }

        block_group.push_back(head);
        block_group.push_back(snk);

        fg->connect(last_node, 0, head, 0);
        fg->connect(head, 0, snk, 0);

        auto opts = schedulers::scheduler_nbt_options::make();
        opts->default_buffer_size = buffer_size;
        auto sched = schedulers::scheduler_nbt::make(opts);


        if (nthreads == 1) {
            sched->add_block_group(block_group, "single thread group");
        }

        auto rt = runtime::make();
        rt->add_scheduler(sched);
        rt->initialize(fg);

        auto t1 = std::chrono::steady_clock::now();

        rt->start();
        rt->wait();

        auto t2 = std::chrono::steady_clock::now();
        auto time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9;

        std::cout << "[PROFILE_TIME]" << time << "[PROFILE_TIME]" << std::endl;
    }
}
