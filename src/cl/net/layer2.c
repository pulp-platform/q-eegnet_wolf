/**
 * @file layer2.c
 * @author Tibor Schneider
 * @date 2020/01/31
 * @brief This file contains the Implementation for the second layer
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

#ifdef PARALLEL

#ifndef NUM_WORKERS
#define NUM_WORKERS 8
#endif

typedef struct
{
    int8_t* p_data;    // pointer to current input image (L1 memory)
    int8_t* p_weight;  // pointer to current weight vector (L1 memory)
    int32_t offset;    // BN offset
    int32_t factor;    // BN factor
    int8_t* p_result; // pointer to result vector (L1 memory)
} _net_layer2_kernel_t;

/**
 * @brief Layer2 kernel
 */
void _net_layer2_kernel(void* args) {

    // get core id
    unsigned int core_id = rt_core_id();

    // extract parameters
    int8_t* _p_data = ((_net_layer2_kernel_t*)args)->p_data;
    int8_t* _p_weight = ((_net_layer2_kernel_t*)args)->p_weight;
    int32_t _offset = ((_net_layer2_kernel_t*)args)->offset;
    int32_t _factor = ((_net_layer2_kernel_t*)args)->factor;
    int8_t* _p_result = ((_net_layer2_kernel_t*)args)->p_result;


#ifdef REORDER_BN
    // in case of reorder bn, compute the relu threshold
    int32_t _threshold = -(_offset >> 3);
#else//REORDER_BN
    // in case of no reorder, factor must be made smaller by 2^3
    _factor = _factor >> 3;
    _offset = _offset >> 3;
#endif//REORDER_BN

    unsigned int _t_out = core_id;

    int8_t* _p_data_iter = _p_data + core_id * 8 * NET_C_ALIGN;
    int8_t* _p_result_iter = _p_result + core_id;

    int32_t _sum, _elem;

    // loop until all elements are computed
    while (_t_out < NET_T8_ALIGN) {

        _sum = 0;

        for (int _t_pool = 0; _t_pool < 8; _t_pool++) {

            // do the dot product
            // we copute the dot product over C_ALIGN instead of C, it is faster and the additional elements are 0
            _elem = func_dotp(_p_data_iter, _p_weight, NET_C_ALIGN);

#ifdef REORDER_BN
            // do the ReLU
            _elem = __MAX(_elem, _threshold);
#else//REORDER_BN
            _elem = (_elem + _offset) / _factor;
            _elem = __MAX(_elem, 0);
#endif//REORDER_BN

            // add the element to the sum
            _sum += _elem;

            // increment data pointer
            _p_data_iter += NET_C_ALIGN;

        }

#ifdef REORDER_BN
        // BN
        _sum = _sum + _offset;
        _sum = _sum / _factor;
#else//REORDER_BN
        // avg pooling division
        _sum = _sum >> 3;
#endif//REORDER_BN
        //clamp
        _sum = __CLIP_R(_sum, 127);
        // write sum back
        *_p_result_iter = (int8_t)_sum;

        // go to the next element
        _t_out += NUM_WORKERS;
        _p_data_iter += (NUM_WORKERS - 1) * 8 * NET_C_ALIGN;
        _p_result_iter += NUM_WORKERS;

    }

    // wait for all workers to finish
    rt_team_barrier();
}

#endif //PARALLEL

/**
 * @brief Execute the 2nd layer
 * 
 * This layer does the following operation on the data:
 * 2. Depthwise convolution in space, with NET_D filters per NET_F1, the same filter for each time sample
 * 4. Apply Batch Normalization
 * 5. Apply ReLU
 * 6. Apply Avg Pooling with kernel (1, 8)
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F1, NET_T, NET_C], aligned to [NET_F1, NET_T, NET_C_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F2, NET_T8] aligned to [NET_F2, NET_T8_ALIGN]
 */
void net_layer2(const int8_t* p_data, int8_t * p_result) {

#ifdef FLIP_LAYERS

#ifdef PARALLEL

#ifdef DMA_STREAM

    /*
     * Parallel implementation with data streaming
     */

    const int8_t* _p_data_iter = p_data;          // iterator over the current image of the input
    const int8_t* _p_result_iter = p_result;      // iterator over the current vector of the output

    rt_dma_copy_t _copy;
    rt_dma_copy_t _data_copy;

    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    int8_t* _p_data_loc_next = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T8_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    if (_p_offset_loc == NULL) {
        printf("Not Enough space on L1 memory");
        return;
    }

    // copy all the weights at once, because copying 6 words would generate too much overhead
    rt_dma_memcpy((unsigned int)net_l2_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    // copy all factors
    rt_dma_memcpy((unsigned int)net_l2_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    // copy all offsets
    rt_dma_memcpy((unsigned int)net_l2_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_wait(&_copy);

    int8_t* _p_weight_loc_iter = _p_weight_loc;  // iterator over the current weights (filter)
    int32_t* _p_factor_loc_iter = _p_factor_loc; // iterator over the current factor
    int32_t* _p_offset_loc_iter = _p_offset_loc; // iterator over the current offset
    int8_t* _p_data_loc_iter;                    // iterator over the current local data row

    int32_t _convert_factor;
    int32_t _convert_offset;
    int32_t _relu_threshold;

    int32_t _elem; // stores the current element, for doing dot product and ReLU
    int32_t _sum;  // stores the sum for the pooling

    // copy the first input
    rt_dma_memcpy((unsigned int)_p_data_iter,
                  (unsigned int)_p_data_loc_next,
                  sizeof(int8_t) * NET_T * NET_C_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_data_copy);

    // loop over all input images
    for (unsigned int _k = 0; _k < NET_F1; _k++) {

        // wait until the first data is present
        rt_dma_wait(&_data_copy);

        //swap pointer to the next with pointer to the current
        uint8_t* _p_tmp = _p_data_loc;
        _p_data_loc = _p_data_loc_next;
        _p_data_loc_next = _p_tmp;

        if (_k < NET_F1 - 1) {

            // go to the next image (next one of F1)
            _p_data_iter += NET_T_ALIGN * NET_C_ALIGN;

            // copy the corresponding input data to local memory
            rt_dma_memcpy((unsigned int)_p_data_iter,
                        (unsigned int)_p_data_loc_next,
                        sizeof(int8_t) * NET_T * NET_C_ALIGN,
                        RT_DMA_DIR_EXT2LOC, 0, &_data_copy);
        }

        // reset the local data iterator
        _p_data_loc_iter = _p_data_loc;
        
        // loop over all output filters for the corresponding input image
        for (unsigned int _i = 0; _i < 2; _i++) {

            // get new convert factors
            _convert_factor = *_p_factor_loc_iter++;
            _convert_offset = *_p_offset_loc_iter++;

            // compute the threshold
            _relu_threshold = -(_convert_offset >> 3);

            // prepare the arguments for the cluster
            _net_layer2_kernel_t args;
            args.p_data = _p_data_loc;
            args.p_weight = _p_weight_loc_iter;
            args.factor = _convert_factor;
            args.offset = _convert_offset;
            args.p_result = _p_result_loc;

            // call the cluster
            rt_team_fork(NUM_WORKERS, _net_layer2_kernel, (void*)(&args));

            // copy the values back to L2 memory
            rt_dma_memcpy((unsigned int)_p_result_iter,
                          (unsigned int)_p_result_loc,
                          sizeof(int8_t) * NET_T8,
                          RT_DMA_DIR_LOC2EXT, 0, &_copy);
            rt_dma_wait(&_copy);

            // go to the next output channel
            _p_result_iter += NET_T8_ALIGN;
            // use the next filter
            _p_weight_loc_iter += NET_L2_WEIGHT_LEN;

        }
    }

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);

#else //DMA_STREAM

    /*
     * Parallel implementation without data streaming
     */

    const int8_t* _p_data_iter = p_data;          // iterator over the current image of the input
    const int8_t* _p_result_iter = p_result;      // iterator over the current vector of the output

    rt_dma_copy_t _copy;

    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T8_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    // copy all the weights at once, because copying 6 words would generate too much overhead
    rt_dma_memcpy((unsigned int)net_l2_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    // copy all factors
    rt_dma_memcpy((unsigned int)net_l2_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    // copy all offsets
    rt_dma_memcpy((unsigned int)net_l2_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_wait(&_copy);

    int8_t* _p_weight_loc_iter = _p_weight_loc;  // iterator over the current weights (filter)
    int32_t* _p_factor_loc_iter = _p_factor_loc; // iterator over the current factor
    int32_t* _p_offset_loc_iter = _p_offset_loc; // iterator over the current offset
    int8_t* _p_data_loc_iter;                    // iterator over the current local data row

    int32_t _convert_factor;
    int32_t _convert_offset;
    int32_t _relu_threshold;

    int32_t _elem; // stores the current element, for doing dot product and ReLU
    int32_t _sum;  // stores the sum for the pooling

    // loop over all input images
    for (unsigned int _k = 0; _k < NET_F1; _k++) {

        // copy the corresponding input data to local memory
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)_p_data_loc,
                      sizeof(int8_t) * NET_T * NET_C_ALIGN,
                      RT_DMA_DIR_EXT2LOC, 0, &_copy);
        rt_dma_wait(&_copy);

        // reset the local data iterator
        _p_data_loc_iter = _p_data_loc;
        
        // loop over all output filters for the corresponding input image
        for (unsigned int _i = 0; _i < 2; _i++) {

            // get new convert factors
            _convert_factor = *_p_factor_loc_iter++;
            _convert_offset = *_p_offset_loc_iter++;

            // compute the threshold
            _relu_threshold = -(_convert_offset >> 3);

            // prepare the arguments for the cluster
            _net_layer2_kernel_t args;
            args.p_data = _p_data_loc;
            args.p_weight = _p_weight_loc_iter;
            args.factor = _convert_factor;
            args.offset = _convert_offset;
            args.p_result = _p_result_loc;

            // call the cluster
            rt_team_fork(NUM_WORKERS, _net_layer2_kernel, (void*)(&args));

            // copy the values back to L2 memory
            rt_dma_memcpy((unsigned int)_p_result_iter,
                          (unsigned int)_p_result_loc,
                          sizeof(int8_t) * NET_T8,
                          RT_DMA_DIR_LOC2EXT, 0, &_copy);
            rt_dma_wait(&_copy);

            // go to the next output channel
            _p_result_iter += NET_T8_ALIGN;
            // use the next filter
            _p_weight_loc_iter += NET_L2_WEIGHT_LEN;

        }

        // go to the next image (next one of F1)
        _p_data_iter += NET_T_ALIGN * NET_C_ALIGN;
    }

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);

#endif //DMA_STREAM

#else //PARALLEL
    
    /*
     * Single Core implementation
     * We compute one output channel (one of F2) at a time.
     */

    const int8_t* _p_data_iter = p_data;          // iterator over the current image of the input
    const int8_t* _p_result_iter = p_result;      // iterator over the current vector of the output

    rt_dma_copy_t _copy;

    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T8_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    // copy all the weights at once, because copying 6 words would generate too much overhead
    rt_dma_memcpy((unsigned int)net_l2_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    // copy all factors
    rt_dma_memcpy((unsigned int)net_l2_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    // copy all offsets
    rt_dma_memcpy((unsigned int)net_l2_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_wait(&_copy);

    int8_t* _p_weight_loc_iter = _p_weight_loc;  // iterator over the current weights (filter)
    int32_t* _p_factor_loc_iter = _p_factor_loc; // iterator over the current factor
    int32_t* _p_offset_loc_iter = _p_offset_loc; // iterator over the current offset
    int8_t* _p_data_loc_iter;                    // iterator over the current local data row
    int8_t* _p_result_loc_iter;                 // iterator over the current temporary result

    int32_t _convert_factor;
    int32_t _convert_offset;
    int32_t _relu_threshold;

    int32_t _elem; // stores the current element, for doing dot product and ReLU
    int32_t _sum;  // stores the sum for the pooling

    // loop over all input images
    for (unsigned int _k = 0; _k < NET_F1; _k++) {

        // copy the corresponding input data to local memory
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)_p_data_loc,
                      sizeof(int8_t) * NET_T * NET_C_ALIGN,
                      RT_DMA_DIR_EXT2LOC, 0, &_copy);
        rt_dma_wait(&_copy);

        // reset the local data iterator
        _p_data_loc_iter = _p_data_loc;
        
        // loop over all output filters for the corresponding input image
        for (unsigned int _i = 0; _i < 2; _i++) {

            // reset the temporary local result iterator
            _p_result_loc_iter = _p_result_loc;

            // reset the local input iterator to point to the first line
            _p_data_loc_iter = _p_data_loc;

            // compute the threshold
            _convert_factor = *_p_factor_loc_iter++;
            _convert_offset = *_p_offset_loc_iter++;

#ifdef REORDER_BN
            _relu_threshold = -(_convert_offset >> 3);
#else//REORDER_BN
            _convert_factor = _convert_factor >> 3;
            _convert_offset = _convert_offset >> 3;
#endif//REORDER_BN

            // loop over all output time samples (after pooling)
            for (unsigned int _t_out = 0; _t_out < NET_T8; _t_out++) {

                // reset the sum
                _sum = 0;

                // loop over all 8 elements in the local neighborhood
                for (unsigned int _t_pool = 0; _t_pool < 8; _t_pool++) {

                    // do the dot product
                    // we copute the dot product over C_ALIGN instead of C, it is faster and the additional elements are 0
                    _elem = func_dotp(_p_data_loc_iter, _p_weight_loc_iter, NET_C_ALIGN);

#ifdef REORDER_BN
                    // do the ReLU
                    _elem = __MAX(_elem, _relu_threshold);
#else//REORDER_BN
                    // do the BN
                    _elem = (_elem + _convert_offset) / _convert_factor;
                    // do the ReLU
                    _elem = __MAX(_elem, 0);
#endif//REORDER_BN

                    // add the element to the sum
                    _sum += _elem;

                    // go to the next input row
                    _p_data_loc_iter += NET_C_ALIGN;
                }

#ifdef REORDER_BN
                // do BN
                _sum = _sum + _convert_offset;
                _sum = _sum / _convert_factor;
#else//REORDER_BN
                // do avg pooling division
                _sum = _sum >> 3;
#endif//REORDER_BN
                // clip
                _sum = __CLIP_R(_sum, 127);
                // write back
                *(_p_result_loc_iter++) = (int8_t)_sum;

            }

            // copy the values back to L2 memory
            rt_dma_memcpy((unsigned int)_p_result_iter,
                          (unsigned int)_p_result_loc,
                          sizeof(int8_t) * NET_T8,
                          RT_DMA_DIR_LOC2EXT, 0, &_copy);
            rt_dma_wait(&_copy);

            // go to the next output channel
            _p_result_iter += NET_T8_ALIGN;
            // use the next filter
            _p_weight_loc_iter += NET_L2_WEIGHT_LEN;

        }

        // go to the next image (next one of F1)
        _p_data_iter += NET_T_ALIGN * NET_C_ALIGN;
    }

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);

#endif //PARALLEL

#else //FLIP_LAYERS

    /*
     * Single Core implementation without flipped input
     * We compute one output channel (one of F2) at a time.
     */

    const int8_t* _p_data_iter = p_data;          // iterator over the current image of the input
    const int8_t* _p_result_iter = p_result;      // iterator over the current vector of the output

    rt_dma_copy_t _copy;

    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_C * NET_T_ALIGN);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T8_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    // copy all the weights at once, because copying 6 words would generate too much overhead
    rt_dma_memcpy((unsigned int)net_l2_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    // copy all factors
    rt_dma_memcpy((unsigned int)net_l2_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    // copy all offsets
    rt_dma_memcpy((unsigned int)net_l2_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_wait(&_copy);

    int8_t* _p_weight_loc_iter = _p_weight_loc;  // iterator over the current weights (filter)
    int32_t* _p_factor_loc_iter = _p_factor_loc; // iterator over the current factor
    int32_t* _p_offset_loc_iter = _p_offset_loc; // iterator over the current offset
    int8_t* _p_data_loc_iter;                    // iterator over the current local data row
    int8_t* _p_result_loc_iter;                 // iterator over the current temporary result

    int32_t _convert_factor;
    int32_t _convert_offset;
    int32_t _relu_threshold;

    int32_t _elem; // stores the current element, for doing dot product and ReLU
    int32_t _sum;  // stores the sum for the pooling

    // loop over all input images
    for (unsigned int _k = 0; _k < NET_F1; _k++) {

        // copy the corresponding input data to local memory
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)_p_data_loc,
                      sizeof(int8_t) * NET_C * NET_T_ALIGN,
                      RT_DMA_DIR_EXT2LOC, 0, &_copy);
        rt_dma_wait(&_copy);

        // reset the local data iterator
        _p_data_loc_iter = _p_data_loc;
        
        // loop over all output filters for the corresponding input image
        for (unsigned int _i = 0; _i < 2; _i++) {

            // reset the temporary local result iterator
            _p_result_loc_iter = _p_result_loc;
            // reset the local input iterator to point to the first line
            _p_data_loc_iter = _p_data_loc;

            // compute the threshold
            _convert_factor = *_p_factor_loc_iter++;
            _convert_offset = *_p_offset_loc_iter++;

#ifdef REORDER_BN
            _relu_threshold = -(_convert_offset >> 3);
#else//REORDER_BN
            _convert_factor = _convert_factor >> 3;
            _convert_offset = _convert_offset >> 3;
#endif//REORDER_BN

            // loop over all output time samples (after pooling)
            for (unsigned int _t_out = 0; _t_out < NET_T8; _t_out++) {

                // reset the sum
                _sum = 0;

                // loop over all 8 elements in the local neighborhood
                for (unsigned int _t_pool = 0; _t_pool < 8; _t_pool++) {

                    // do the dot product
                    _elem = func_dotp_slow(_p_data_loc_iter, NET_T_ALIGN, _p_weight_loc_iter, 1, NET_C);

#ifdef REORDER_BN
                    // do the ReLU
                    _elem = __MAX(_elem, _relu_threshold);
#else//REORDER_BN
                    // do the BN
                    _elem = (_elem + _convert_offset) / _convert_factor;
                    // do the ReLU
                    _elem = __MAX(_elem, 0);
#endif//REORDER_BN

                    // add the element to the sum
                    _sum += _elem;

                    // go to the next input row
                    _p_data_loc_iter += 1;
                }

#ifdef REORDER_BN
                // do BN
                _sum = _sum + _convert_offset;
                _sum = _sum / _convert_factor;
#else//REORDER_BN
                // do avg pooling division
                _sum = _sum >> 3;
#endif//REORDER_BN
                // clip
                _sum = __CLIP_R(_sum, 127);
                // write back
                *(_p_result_loc_iter++) = (int8_t)_sum;

            }

            // copy the values back to L2 memory
            rt_dma_memcpy((unsigned int)_p_result_iter,
                          (unsigned int)_p_result_loc,
                          sizeof(int8_t) * NET_T8,
                          RT_DMA_DIR_LOC2EXT, 0, &_copy);
            rt_dma_wait(&_copy);

            // go to the next output channel
            _p_result_iter += NET_T8_ALIGN;
            // use the next filter
            _p_weight_loc_iter += NET_L2_WEIGHT_LEN;

        }

        // go to the next image (next one of F1)
        _p_data_iter += NET_C * NET_T_ALIGN;
    }

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_C * NET_T_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);


#endif //FLIP LAYERS

}
