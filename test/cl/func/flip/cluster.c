#include "stdio.h"
#include "rt/rt_api.h"
#include "test_stimuli.h"
#include "../../../../src/cl/func/functional.h"

RT_CL_DATA static int8_t* p_x_l1;
RT_CL_DATA static int8_t* p_y_l1;
RT_CL_DATA static int8_t* p_exp_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);
    
    func_flip_2d_axis(p_x_l1, OUTER_LEN, INNER_LEN, p_y_l1);

    rt_perf_stop(perf);

    int success = 0;
    for (int i = 0; i < INNER_LEN * OUTER_LEN_ALIGN; i++) {
        if (p_exp_l1[i] != p_y_l1[i]) {
            int outer = i / OUTER_LEN_ALIGN;
            int inner = i % OUTER_LEN_ALIGN;
            printf("Error at: %d,%d: acq=%d, exp=%d\n", outer, inner, p_y_l1[i], p_exp_l1[i]);
            success = 1;
        }
    }

    return success;
}


int do_bench_par(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    func_flip_2d_axis_par(p_x_l1, OUTER_LEN, INNER_LEN, p_y_l1);

    rt_perf_stop(perf);

    int success = 0;
    for (int i = 0; i < INNER_LEN * OUTER_LEN_ALIGN; i++) {
        if (p_exp_l1[i] != p_y_l1[i]) {
            int outer = i / OUTER_LEN_ALIGN;
            int inner = i % OUTER_LEN_ALIGN;
            printf("Error at: %d,%d: acq=%d, exp=%d\n", outer, inner, p_y_l1[i], p_exp_l1[i]);
            success = 1;
        }
    }

    return success;
}

void cluster_entry(void* arg) {

    // allocate memory
    p_x_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vec_x));
    p_y_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vec_exp));
    p_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vec_exp));

    if (p_exp_l1 == NULL) {
        printf("Not enough memory!\n");
        return;
    }

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)vec_x, (unsigned int)p_x_l1, sizeof(vec_x), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);
    rt_dma_memcpy((unsigned int)vec_exp, (unsigned int)p_exp_l1, sizeof(vec_exp), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    int result;

    // test without bias
    result = do_bench(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));

    // print the results
    if (result == 0) {
        printf("## normal: result: OK\n");
    } else {
        printf("## normal: result: FAIL\n");
    }
    printf("## normal: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## normal: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));

    // test without bias
    result = do_bench_par(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));

    // print the results
    if (result == 0) {
        printf("## parallel: result: OK\n");
    } else {
        printf("## parallel: result: FAIL\n");
    }
    printf("## parallel: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## parallel: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));
}
