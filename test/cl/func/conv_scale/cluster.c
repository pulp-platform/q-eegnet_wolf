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

RT_CL_DATA static int8_t* pA_l1;
RT_CL_DATA static int8_t* pB_l1;
RT_CL_DATA static int8_t* pRes_l1;
RT_CL_DATA static int8_t* pExp_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);
    
    func_conv_scale(pA_l1, LENGTH_A, pB_l1, LENGTH_B, FACTOR, OFFSET, pRes_l1);

    rt_perf_stop(perf);

    int success = 0;
    for (int i = 0; i < LENGTH_RES; i++) {
        if (pRes_l1[i] != pExp_l1[i]) {
            success = 1;
            printf("at %d: acq=%d, exp=%d\n", i, pRes_l1[i], pExp_l1[i]);
        }
    }

    return success;
}

void cluster_entry(void* arg) {

    // allocate memory
    pA_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vecA));
    pB_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vecB));
    pRes_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vecExp));
    pExp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(vecExp));

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)vecA, (unsigned int)pA_l1, sizeof(vecA), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);
    rt_dma_memcpy((unsigned int)vecB, (unsigned int)pB_l1, sizeof(vecB), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);
    rt_dma_memcpy((unsigned int)vecExp, (unsigned int)pExp_l1, sizeof(vecExp), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);

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
