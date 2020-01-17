#include "stdio.h"
#include "rt/rt_api.h"
#include "plp_math.h"
#include "test_stimuli.h"
// TODO include the functions which should be tested

// TODO prepare arrays
RT_CL_DATA static int8_t* pA_l1;
RT_CL_DATA static int8_t* pB_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);
    
    // TODO do the actual test here

    rt_perf_stop(perf);

    return result - EXP_RESULT;
}

void cluster_entry(void* arg) {

    // allocate memory
    // TODO
    pA_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vecA));
    pB_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vecB));

    // copy memory
    // TODO
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)vecA, (unsigned int)pA_l1, sizeof(vecA), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);
    rt_dma_memcpy((unsigned int)vecB, (unsigned int)pB_l1, sizeof(vecB), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    int result;

    for (int i = 0; i < 10; i++) {
        result = do_bench(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));
    }

    // print the results
    if (result == 0) {
        printf("## 1: result: OK\n");
    } else {
        printf("## 1: result: FAIL\n");
    }
    printf("## 1: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## 1: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));
}
