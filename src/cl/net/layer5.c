/**
 * @file layer5.c
 * @author Tibor Schneider
 * @date 2020/01/31
 * @brief This file contains the Implementation for the fifth layer
 */

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


#include "rt/rt_api.h"
#include "layers.h"
#include "net.h"
#include "../func/functional.h"

/**
 * @brief Execute the 5th layer
 * 
 * This layer does the following operation on the data:
 * 1. Apply linear layer
 *
 * @param p_data Pointer to the input data, of shape [NET_F2, NET_T64], aligned to [NET_F2, NET_T64_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_N]
 */
void net_layer5(const int8_t* p_data, int8_t * p_result) {

    // keep the entire input vector in local memory, but only one weight vector (of the 4)

    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_N);
    int32_t* _p_tmp_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_N);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
    int8_t* _p_bias_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_N);

    rt_dma_copy_t _copy;

    // copy all the data at once
    rt_dma_memcpy((unsigned int)p_data,
                  (unsigned int)_p_data_loc,
                  sizeof(int8_t) * NET_F2 * NET_T64_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    // copy the bias vector (simply one word)
    *((int32_t*)_p_bias_loc) = *((int32_t*)net_l5_bias);

    // prepare the weight iterator
    const int8_t* _p_weight_iter = net_l5_weight;
    int8_t* _p_bias_loc_iter = _p_bias_loc;
    int32_t* _p_tmp_result_loc_iter = _p_tmp_result_loc;

    // loop over all output elements
    for (unsigned int _n = 0; _n < NET_N; _n++) {

        // load weights
        rt_dma_memcpy((unsigned int)_p_weight_iter,
                      (unsigned int)_p_weight_loc,
                      sizeof(int8_t) * NET_F2 * NET_T64_ALIGN,
                      RT_DMA_DIR_EXT2LOC, 0, &_copy);
        rt_dma_wait(&_copy);

        // we multiply the aligned vectors here. It will be faster, and the weight vector has zeros at the aligned positions
        *(_p_tmp_result_loc_iter++) = func_dotp(_p_data_loc, _p_weight_loc, NET_F2 * NET_T64_ALIGN) + (*_p_bias_loc_iter++);

        // go to the next iteration
        _p_weight_iter += NET_F2 * NET_T64_ALIGN;
    }


    // transform the vector
    func_transform_32to8(_p_tmp_result_loc, NET_N, NET_L5_FACTOR, 1, _p_result_loc);

    // copy the data back (one word, do not use DMA)
    *((int32_t*)p_result) = *((int32_t*)_p_result_loc);

    // free the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_N);
    rt_free(RT_ALLOC_CL_DATA, _p_tmp_result_loc, sizeof(int32_t) * NET_N);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_bias_loc, sizeof(int8_t) * NET_N);

}
