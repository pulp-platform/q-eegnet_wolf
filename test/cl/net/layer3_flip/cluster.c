#include "stdio.h"
#include "rt/rt_api.h"
#include "test_stimuli.h"
#include "../../../../src/cl/func/functional.h"
#include "../../../../src/cl/net/net.h"
#include "../../../../src/cl/net/layers.h"

int do_bench(rt_perf_t* perf, int events) {

    printf("Compute layer...\n");

    //setup performance measurement
    rt_perf_conf(perf, events);

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    net_layer3_flip_inplace(x_vec);

    rt_perf_stop(perf);

    // print the results back to stdout, so that it can be checked there
    int error = 0;
    for (int k = 0; k < NET_F2; k++) {
        for (int t = 0; t < NET_T8; t++) {
            if (x_vec[k + t * NET_F2] !=
                y_exp[k + t * NET_F2]) {
                error = 1;
            }
        }
    }

    return error;
}

void cluster_entry(void* arg) {

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    int result;

    result = do_bench(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));

    // print the results
    if (result == 0) {
        printf("## 1: result: OK\n");
    } else {
        printf("## 1: result: FAIL\n");
    }
    printf("## 1: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## 1: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));
}
