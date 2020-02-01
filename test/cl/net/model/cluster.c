#include "stdio.h"
#include "rt/rt_api.h"
#include "test_stimuli.h"
#include "../../../../src/cl/net/net.h"
#include "../../../../src/cl/net/model.h"

int do_bench(rt_perf_t* perf, int events) {

    // allocate result memory
    int8_t * p_output = rt_alloc(RT_ALLOC_FC_DATA, sizeof(int8_t) * NET_N);

    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);
    
    net_model_compute(x_vec, p_output);

    rt_perf_stop(perf);

    int num_err = 0;
    for (int n = 0; n < NET_N; n++) {
        if (p_output[n] != y_exp_vec[n]) {
            num_err++;
        }
    }

    // free memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*) p_output, sizeof(int8_t) * NET_N);

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
}
