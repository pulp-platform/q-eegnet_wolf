#include "stdio.h"
#include "rt/rt_api.h"
#include "test_stimuli.h"
#include "../../../../src/cl/net/net.h"
#include "../../../../src/cl/net/layers.h"

int do_bench(rt_perf_t* perf, int events) {

    // allocate result memory
    int8_t * p_output = rt_alloc(RT_ALLOC_FC_DATA, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);

    printf("Compute layer...\n");
    
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);
    
    net_layer4(x_vec, p_output);

    rt_perf_stop(perf);

    printf("Compare results...\n");

    int num_err = 0;
    for (int k = 0; k < NET_F2; k++) {
        for (int t = 0; t < NET_T64; t++) {
            if (p_output[k * NET_T64_ALIGN + t] != y_exp_vec[k * NET_T64_ALIGN + t]) {
                num_err++;
            }
        }
    }

    // free memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*) p_output, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    return num_err;
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
    printf("## 1: diff: %d\n", result);
}
