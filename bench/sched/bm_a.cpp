#include <chrono>
#include <iostream>
#include <thread>

#include <gnuradio/bench/copy.h>
#include <gnuradio/blocks/null_sink.h>
#include <gnuradio/blocks/null_source.h>
#include <gnuradio/flowgraph.h>
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
    bool use_memcpy = true;
    bool rt_prio = false;
    size_t buffer_size = 32768;

    std::vector<unsigned int> cpu_affinity;

    CLI::App app{ "App description" };

    // app.add_option("-h,--help", "display help");
    app.add_option("--samples", samples, "Number of Samples");
    app.add_option("--stages", stages, "Number of copy blocks");
    app.add_option("--buffer_size", buffer_size, "Default Buffer Size");
    app.add_option("--nthreads", nthreads, "Number of threads (0: tpb)");
    app.add_option("--use_memcpy", use_memcpy, "use memcpy for copy");
    app.add_flag("--rt_prio", rt_prio, "Enable Real-time priority");

    CLI11_PARSE(app, argc, argv);

    if (rt_prio && gr::enable_realtime_scheduling() != rt_status_t::OK) {
        std::cout << "Error: failed to enable real-time scheduling." << std::endl;
    }

    {
        auto src = blocks::null_source::make({ 1, sizeof(float) });
        auto head =
            streamops::head::make_cpu({ samples, sizeof(float)  });
        auto snk = blocks::null_sink::make({ 1, sizeof(float)  });
        std::vector<gr::bench::copy<float>::sptr> copy_blks(stages*3);
        for (unsigned int i = 0; i < stages; i++) {
            copy_blks[i*3] = bench::copy<float>::make({ use_memcpy });
            copy_blks[i*3+1] = bench::copy<float>::make({ use_memcpy});
            copy_blks[i*3+2] = bench::copy<float>::make({ use_memcpy });
        }
        auto fg = flowgraph::make();



        gr::node_sptr last_node = src;       

        for (unsigned int i = 0; i < stages; i++) {
            fg->connect(last_node, 0, copy_blks[3*i], 0)->set_custom_buffer(
                BUFFER_CPU_VMCIRC_ARGS->set_max_buffer_read(128)
            );
            fg->connect(copy_blks[3*i], 0, copy_blks[3*i+1], 0)->set_custom_buffer(
                BUFFER_CPU_VMCIRC_ARGS->set_max_buffer_read(1024)
            );
            fg->connect(copy_blks[3*i+1], 0, copy_blks[3*i+2], 0)->set_custom_buffer(
                BUFFER_CPU_VMCIRC_ARGS->set_max_buffer_read(128)->set_min_buffer_read(32)
            );
            // fg->connect(last_node, 0, copy_blks[3*i], 0);
            // fg->connect(copy_blks[3*i], 0, copy_blks[3*i+1], 0);
            // fg->connect(copy_blks[3*i+1], 0, copy_blks[3*i+2], 0);
            last_node = copy_blks[3*i+2];
        }
        fg->connect(last_node, 0, head, 0);
        fg->connect(head, 0, snk, 0);

        auto opts = schedulers::scheduler_nbt_options::make();
        opts->default_buffer_size = buffer_size;
        auto sched = schedulers::scheduler_nbt::make(opts);


        if (nthreads == 1) {
            std::vector<block_sptr> block_group;

            block_group.push_back(src);



            for (size_t j = 0; j < stages * 3; j++) {
                block_group.push_back(copy_blks[j]);
            }

            block_group.push_back(head);
            block_group.push_back(snk);

            // sched->add_block_group(block_group);

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
