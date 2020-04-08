/*
 * Copyright (C) 2020 ETH Zurich. All rights reserved.
 *
 * Author: Tibor Schneider, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "stdio.h"
#include "rt/rt_api.h"
#include "test_stimuli.h"
#include "../../../../src/cl/func/functional.h"

RT_CL_DATA static int32_t* p_x_l1;
RT_CL_DATA static int8_t* p_y_l1;
RT_CL_DATA static int8_t* p_exp_l1;
RT_CL_DATA static int8_t* p_exp_bias_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);
    
    func_transform_32to8(p_x_l1, LENGTH, div_factor, 1, p_y_l1);

    rt_perf_stop(perf);

    int success = 0;
    for (int i = 0; i < LENGTH; i++) {
        if (p_y_l1[i] != p_exp_l1[i]) {
            success = 1;
        }
    }

    return success;
}

int do_bench_bias(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);
    
    func_transform_32to8_bias(p_x_l1, LENGTH, div_factor, bias, 1, p_y_l1);

    rt_perf_stop(perf);

    int success = 0;
    for (int i = 0; i < LENGTH; i++) {
        if (p_y_l1[i] != p_exp_bias_l1[i]) {
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
    p_exp_bias_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vec_exp_bias));

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)vec_x, (unsigned int)p_x_l1, sizeof(vec_x), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);
    rt_dma_memcpy((unsigned int)vec_exp, (unsigned int)p_exp_l1, sizeof(vec_exp), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);
    rt_dma_memcpy((unsigned int)vec_exp_bias, (unsigned int)p_exp_bias_l1, sizeof(vec_exp_bias), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    int result;

    // test without bias
    for (int i = 0; i < 10; i++) {
        result = do_bench(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR | 1<<RT_PERF_LD_STALL));
    }

    // print the results
    if (result == 0) {
        printf("## factor: result: OK\n");
    } else {
        printf("## factor: result: FAIL\n");
    }
    printf("## factor: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## factor: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));
    printf("## factor: load stalls: %d\n", rt_perf_read(RT_PERF_LD_STALL));

    // test with bias
    for (int i = 0; i < 10; i++) {
        result = do_bench_bias(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR | 1<<RT_PERF_LD_STALL));
    }

    // print the results
    if (result == 0) {
        printf("## factor+bias: result: OK\n");
    } else {
        printf("## factor+bias: result: FAIL\n");
    }
    printf("## factor+bias: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## factor+bias: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));
    printf("## factor+bias: load stalls: %d\n", rt_perf_read(RT_PERF_LD_STALL));
}
