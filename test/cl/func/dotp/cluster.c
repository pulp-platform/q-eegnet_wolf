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
//#include "plp_math.h"

RT_CL_DATA static int8_t* p_a_l1;
RT_CL_DATA static int8_t* p_b_l1;

int do_bench(rt_perf_t* perf, int events) {

    int32_t acq_result;
    
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);
    
    acq_result = func_dotp(p_a_l1, p_b_l1, LENGTH);
    //plp_dot_prod_i8v_xpulpv2(p_a_l1, p_b_l1, LENGTH, &acq_result);

    rt_perf_stop(perf);

    int error = acq_result == EXP_RESULT ? 0 : 1;

    return error;
}

void cluster_entry(void* arg) {

    // allocate memory
    p_a_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vec_a));
    p_b_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vec_b));

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)vec_a, (unsigned int)p_a_l1, sizeof(vec_a), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);
    rt_dma_memcpy((unsigned int)vec_b, (unsigned int)p_b_l1, sizeof(vec_b), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    int result;

    // test without bias
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
