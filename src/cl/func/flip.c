/**
 * @file functional.h
 * @author Tibor Schneider
 * @date 2020/01/30
 * @brief This file contains the definitions of the 2d flip function
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
#include "functional.h"

#define _SHUFFLE_MASK_1_1 (v4s){0,4,1,5}
#define _SHUFFLE_MASK_1_2 (v4s){2,6,3,7}
#define _SHUFFLE_MASK_2_1 (v4s){0,1,4,5}
#define _SHUFFLE_MASK_2_2 (v4s){2,3,6,7}

#ifndef NUM_WORKERS
#define NUM_WORKERS 8
#endif

/**
 * @brief Local function to flip inner and outer dimension of a 2d axis.
 * 
 * The inner dimension of the input array must be 4 Bytes aligned. This means, that every
 * row starts at an aligned address. However, the parameter inner_len must be given as the unaligned
 * number of elements.
 *
 * The inner dimension of the output array will also be aligned to 4 Bytes. The output vector must
 * already be allocated in local (L1) memory. The size will be: inner_len * ((outer_len + 3) / 4) * 4
 *
 * The data must be present in local L1 memory
 * 
 * @param p_in Pointer to the input vector on L1 memory, of shape outer_len * ((inner_len + 3) / 4) * 4
 * @param outer_len Length of the outer dimension, not necessarily aligned
 * @param inner_len Actual length of the inner dimension, not necessarily aligned
 * @param chunk_width Width of the current chunk being processed
 * @param p_res Pointer to the output vector on L1 memory, must already be allocated.
 */
void _func_flip_2d_axis_chunk(const int8_t* p_in,
                              unsigned int outer_len,
                              unsigned int inner_len,
                              unsigned int chunk_width,
                              int8_t* p_res);


typedef struct {
    const int8_t* p_in;
    unsigned int outer_len;
    unsigned int inner_len;
    int8_t* p_res;
} _func_flip_2d_axis_kernel_t;

void _func_flip_2d_axis_kernel(void* args);

/**
 * @brief Flip inner and outer dimension of a 2d axis.
 * 
 * The inner dimension of the input array must be 4 Bytes aligned. This means, that every
 * row starts at an aligned address. However, the parameter inner_len must be given as the unaligned
 * number of elements.
 *
 * The inner dimension of the output array will also be aligned to 4 Bytes. The output vector must
 * already be allocated in local (L1) memory. The size will be: inner_len * ((outer_len + 3) / 4) * 4
 *
 * The data must be present in local L1 memory
 * 
 * @param p_in Pointer to the input vector on L1 memory, of shape outer_len * ((inner_len + 3) / 4) * 4
 * @param outer_len Length of the outer dimension, not necessarily aligned
 * @param inner_len Actual length of the inner dimension, not necessarily aligned
 * @param p_res Pointer to the output vector on L1 memory, must already be allocated.
 */
void func_flip_2d_axis(const int8_t* p_in,
                       unsigned int outer_len,
                       unsigned int inner_len,
                       int8_t * p_res) {

    _func_flip_2d_axis_chunk(p_in, outer_len, inner_len, inner_len, p_res);

}

/**
 * @brief Parallel function to flip inner and outer dimension of a 2d axis.
 *
 * The inner dimension of the input array must be 4 Bytes aligned. This means, that every
 * row starts at an aligned address. However, the parameter inner_len must be given as the unaligned
 * number of elements.
 *
 * The inner dimension of the output array will also be aligned to 4 Bytes. The output vector must
 * already be allocated in local (L1) memory. The size will be: inner_len * ((outer_len + 3) / 4) * 4
 *
 * The data must be present in local L1 memory
 *
 * @param p_in Pointer to the input vector on L1 memory, of shape outer_len * ((inner_len + 3) / 4) * 4
 * @param outer_len Length of the outer dimension, not necessarily aligned
 * @param inner_len Actual length of the inner dimension, not necessarily aligned
 * @param p_res Pointer to the output vector on L1 memory, must already be allocated.
 */
void func_flip_2d_axis_par(const int8_t* p_in,
                           unsigned int outer_len,
                           unsigned int inner_len,
                           int8_t* p_res) {

    // call the kernel
    _func_flip_2d_axis_kernel_t args;
    args.p_in = p_in;
    args.outer_len = outer_len;
    args.inner_len = inner_len;
    args.p_res = p_res;

    rt_team_fork(NUM_WORKERS, _func_flip_2d_axis_kernel, &args);

}

void _func_flip_2d_axis_kernel(void* args) {

    const int8_t* p_in = ((_func_flip_2d_axis_kernel_t*)args)->p_in;
    unsigned int outer_len = ((_func_flip_2d_axis_kernel_t*)args)->outer_len;
    unsigned int inner_len = ((_func_flip_2d_axis_kernel_t*)args)->inner_len;
    int8_t* p_res = ((_func_flip_2d_axis_kernel_t*)args)->p_res;

    unsigned int _core_id = rt_core_id();

    unsigned int _chunk_width = inner_len / NUM_WORKERS;
    unsigned int _outer_len_aligned = ((outer_len + 3) / 4) * 4;
    // change the p_in to point to the start of the chunk
    p_in += _core_id * _chunk_width;
    p_res += _core_id * _chunk_width * _outer_len_aligned;

    // if this is the last core, update the chunk width
    if (_core_id == NUM_WORKERS - 1) {
        _chunk_width = inner_len - (NUM_WORKERS - 1) * _chunk_width;
    }

    _func_flip_2d_axis_chunk(p_in, outer_len, inner_len, _chunk_width, p_res);

    // wait until all workers are finished
    rt_team_barrier();
}


void _func_flip_2d_axis_chunk(const int8_t* p_in,
                              unsigned int outer_len,
                              unsigned int inner_len,
                              unsigned int chunk_width,
                              int8_t* p_res) { 

    unsigned int _inner_len_aligned = ((inner_len + 3) / 4) * 4;
    unsigned int _outer_len_aligned = ((outer_len + 3) / 4) * 4;

    // blocks are 4 columns inside the chunk
    unsigned int _num_blk = chunk_width / 4;
    unsigned int _rem_blk = chunk_width % 4;
    const int8_t* _p_in_block_iter = p_in;
    int8_t* _p_out_block_iter = p_res;

    // parts are 4 rows inside the block
    unsigned int _num_part;
    unsigned int _rem_part = outer_len % 4;
    const int8_t* _p_in_part_iter;
    int8_t* _p_out_part_iter;

    // data used in the inner most loop
    v4s _x0, _x1, _x2, _x3; //input
    v4s _y0, _y1, _y2, _y3; //output
    v4s _tmp1, _tmp2;       //temporary shuffle result
    v4s _zero = (v4s){0, 0, 0, 0};

    while (_num_blk > 0) {

        /*
         * In the block, iterate over a part of 4 elements and combine them
         */

        _num_part = outer_len / 4;
        _p_in_part_iter = _p_in_block_iter;
        _p_out_part_iter = _p_out_block_iter;

        while (_num_part > 0) {

            // load the input
            _x0 = *((v4s*)(_p_in_part_iter + 0 * _inner_len_aligned));
            _x1 = *((v4s*)(_p_in_part_iter + 1 * _inner_len_aligned));
            _x2 = *((v4s*)(_p_in_part_iter + 2 * _inner_len_aligned));
            _x3 = *((v4s*)(_p_in_part_iter + 3 * _inner_len_aligned));

            // compute _y0, _y1
            _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_1);
            _tmp2 = __builtin_shuffle(_x2, _x3, _SHUFFLE_MASK_1_1);
            _y0 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_1);
            _y1 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_2);

            // compute _y2, _y3
            _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_2);
            _tmp2 = __builtin_shuffle(_x2, _x3, _SHUFFLE_MASK_1_2);
            _y2 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_1);
            _y3 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_2);

            // store the result
            *((int32_t*)(_p_out_part_iter + 0 * _outer_len_aligned)) = (int32_t)_y0;
            *((int32_t*)(_p_out_part_iter + 1 * _outer_len_aligned)) = (int32_t)_y1;
            *((int32_t*)(_p_out_part_iter + 2 * _outer_len_aligned)) = (int32_t)_y2;
            *((int32_t*)(_p_out_part_iter + 3 * _outer_len_aligned)) = (int32_t)_y3;

            // go to the next part in the block
            _p_in_part_iter += 4 * _inner_len_aligned;
            _p_out_part_iter += 4;
            _num_part--;

        }

        // do the remaining incomplete parts
        if (_rem_part == 1) {

            _x0 = *((v4s*)(_p_in_part_iter + 0 * _inner_len_aligned));

            _y0 = __builtin_shuffle(_x0, _zero, (v4s){0, 5, 6, 7});
            _y1 = __builtin_shuffle(_x0, _zero, (v4s){1, 5, 6, 7});
            _y2 = __builtin_shuffle(_x0, _zero, (v4s){2, 5, 6, 7});
            _y3 = __builtin_shuffle(_x0, _zero, (v4s){3, 5, 6, 7});

            // store the result
            *((int32_t*)(_p_out_part_iter + 0 * _outer_len_aligned)) = (int32_t)_y0;
            *((int32_t*)(_p_out_part_iter + 1 * _outer_len_aligned)) = (int32_t)_y1;
            *((int32_t*)(_p_out_part_iter + 2 * _outer_len_aligned)) = (int32_t)_y2;
            *((int32_t*)(_p_out_part_iter + 3 * _outer_len_aligned)) = (int32_t)_y3;

        } else if (_rem_part == 2) {

            _x0 = *((v4s*)(_p_in_part_iter + 0 * _inner_len_aligned));
            _x1 = *((v4s*)(_p_in_part_iter + 1 * _inner_len_aligned));

            _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_1);
            _tmp2 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_2);

            _y0 = __builtin_shuffle(_tmp1, _zero, _SHUFFLE_MASK_2_1);
            _y1 = __builtin_shuffle(_tmp1, _zero, _SHUFFLE_MASK_2_2);
            _y2 = __builtin_shuffle(_tmp2, _zero, _SHUFFLE_MASK_2_1);
            _y3 = __builtin_shuffle(_tmp2, _zero, _SHUFFLE_MASK_2_2);

            // store the result
            *((int32_t*)(_p_out_part_iter + 0 * _outer_len_aligned)) = (int32_t)_y0;
            *((int32_t*)(_p_out_part_iter + 1 * _outer_len_aligned)) = (int32_t)_y1;
            *((int32_t*)(_p_out_part_iter + 2 * _outer_len_aligned)) = (int32_t)_y2;
            *((int32_t*)(_p_out_part_iter + 3 * _outer_len_aligned)) = (int32_t)_y3;

        } else if (_rem_part == 3) {

            _x0 = *((v4s*)(_p_in_part_iter + 0 * _inner_len_aligned));
            _x1 = *((v4s*)(_p_in_part_iter + 1 * _inner_len_aligned));
            _x2 = *((v4s*)(_p_in_part_iter + 2 * _inner_len_aligned));

            // compute _y0, _y1
            _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_1);
            _tmp2 = __builtin_shuffle(_x2, _zero, _SHUFFLE_MASK_1_1);
            _y0 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_1);
            _y1 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_2);

            // compute _y2, _y3
            _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_2);
            _tmp2 = __builtin_shuffle(_x2, _zero, _SHUFFLE_MASK_1_2);
            _y2 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_1);
            _y3 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_2);

            // store the result
            *((int32_t*)(_p_out_part_iter + 0 * _outer_len_aligned)) = (int32_t)_y0;
            *((int32_t*)(_p_out_part_iter + 1 * _outer_len_aligned)) = (int32_t)_y1;
            *((int32_t*)(_p_out_part_iter + 2 * _outer_len_aligned)) = (int32_t)_y2;
            *((int32_t*)(_p_out_part_iter + 3 * _outer_len_aligned)) = (int32_t)_y3;

        }

        // go to the next block in the chunk
        _p_in_block_iter += 4;
        _p_out_block_iter += 4 * _outer_len_aligned; //outer_len is the inner dimension of the output
        _num_blk--;
    }

    /*
     * Here, we compute the incomplete bock.
     * We compute the normal values as usual, but just store fewer values.
     * The loop over all full parts is split into three cases (rem_blk = 1 | 2 | 3), but
     * the last incomplete part is done with multiple if conditions. 
     */

    if (_rem_blk >= 1) {
        _num_part = outer_len / 4;
        _p_in_part_iter = _p_in_block_iter;
        _p_out_part_iter = _p_out_block_iter;

        // we do 3 different loops, for each case of _rem_blk.
        if (_rem_blk == 1) {

            while (_num_part > 0) {

                // load the input
                _x0 = *((v4s*)(_p_in_part_iter + 0 * _inner_len_aligned));
                _x1 = *((v4s*)(_p_in_part_iter + 1 * _inner_len_aligned));
                _x2 = *((v4s*)(_p_in_part_iter + 2 * _inner_len_aligned));
                _x3 = *((v4s*)(_p_in_part_iter + 3 * _inner_len_aligned));

                // compute _y0, _y1
                _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_1);
                _tmp2 = __builtin_shuffle(_x2, _x3, _SHUFFLE_MASK_1_1);
                _y0 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_1);

                // store the result
                *((int32_t*)(_p_out_part_iter + 0 * _outer_len_aligned)) = (int32_t)_y0;

                // go to the next part in the block
                _p_in_part_iter += 4 * _inner_len_aligned;
                _p_out_part_iter += 4;
                _num_part--;

            }

        } else if (_rem_blk == 2) {

            while (_num_part > 0) {

                // load the input
                _x0 = *((v4s*)(_p_in_part_iter + 0 * _inner_len_aligned));
                _x1 = *((v4s*)(_p_in_part_iter + 1 * _inner_len_aligned));
                _x2 = *((v4s*)(_p_in_part_iter + 2 * _inner_len_aligned));
                _x3 = *((v4s*)(_p_in_part_iter + 3 * _inner_len_aligned));

                // compute _y0, _y1
                _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_1);
                _tmp2 = __builtin_shuffle(_x2, _x3, _SHUFFLE_MASK_1_1);
                _y0 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_1);
                _y1 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_2);

                // store the result
                *((int32_t*)(_p_out_part_iter + 0 * _outer_len_aligned)) = (int32_t)_y0;
                *((int32_t*)(_p_out_part_iter + 1 * _outer_len_aligned)) = (int32_t)_y1;

                // go to the next part in the block
                _p_in_part_iter += 4 * _inner_len_aligned;
                _p_out_part_iter += 4;
                _num_part--;

            }

        } else { //_rem_blk == 3

            while (_num_part > 0) {

                // load the input
                _x0 = *((v4s*)(_p_in_part_iter + 0 * _inner_len_aligned));
                _x1 = *((v4s*)(_p_in_part_iter + 1 * _inner_len_aligned));
                _x2 = *((v4s*)(_p_in_part_iter + 2 * _inner_len_aligned));
                _x3 = *((v4s*)(_p_in_part_iter + 3 * _inner_len_aligned));

                // compute _y0, _y1
                _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_1);
                _tmp2 = __builtin_shuffle(_x2, _x3, _SHUFFLE_MASK_1_1);
                _y0 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_1);
                _y1 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_2);

                // compute _y2, _y3
                _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_2);
                _tmp2 = __builtin_shuffle(_x2, _x3, _SHUFFLE_MASK_1_2);
                _y2 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_1);

                // store the result
                *((int32_t*)(_p_out_part_iter + 0 * _outer_len_aligned)) = (int32_t)_y0;
                *((int32_t*)(_p_out_part_iter + 1 * _outer_len_aligned)) = (int32_t)_y1;
                *((int32_t*)(_p_out_part_iter + 2 * _outer_len_aligned)) = (int32_t)_y2;

                // go to the next part in the block
                _p_in_part_iter += 4 * _inner_len_aligned;
                _p_out_part_iter += 4;
                _num_part--;

            }

        }

        // do the remaining incomplete parts of the incomplete block

        if (_rem_part == 1) {

            // load the data
            _x0 = *((v4s*)(_p_in_part_iter + 0 * _inner_len_aligned));

            // if (_rem_blk >= 1), always true
            _y0 = __builtin_shuffle(_x0, _zero, (v4s){0, 5, 6, 7});
            *((int32_t*)(_p_out_part_iter + 0 * _outer_len_aligned)) = (int32_t)_y0;

            if (_rem_blk >= 2) {
                _y1 = __builtin_shuffle(_x0, _zero, (v4s){1, 5, 6, 7});
                *((int32_t*)(_p_out_part_iter + 1 * _outer_len_aligned)) = (int32_t)_y1;
            }
            if (_rem_blk >= 3) {
                _y2 = __builtin_shuffle(_x0, _zero, (v4s){2, 5, 6, 7});
                *((int32_t*)(_p_out_part_iter + 2 * _outer_len_aligned)) = (int32_t)_y2;
            }

        } else if (_rem_part == 2) {

            _x0 = *((v4s*)(_p_in_part_iter + 0 * _inner_len_aligned));
            _x1 = *((v4s*)(_p_in_part_iter + 1 * _inner_len_aligned));

            _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_1);
            _tmp2 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_2);

            // if (_rem_blk >= 1), always true
            _y0 = __builtin_shuffle(_tmp1, _zero, _SHUFFLE_MASK_2_1);
            *((int32_t*)(_p_out_part_iter + 0 * _outer_len_aligned)) = (int32_t)_y0;

            if (_rem_blk >= 2) {
                _y1 = __builtin_shuffle(_tmp1, _zero, _SHUFFLE_MASK_2_2);
                *((int32_t*)(_p_out_part_iter + 1 * _outer_len_aligned)) = (int32_t)_y1;
            }
            if (_rem_blk >= 3) {
                _y2 = __builtin_shuffle(_tmp2, _zero, _SHUFFLE_MASK_2_1);
                *((int32_t*)(_p_out_part_iter + 2 * _outer_len_aligned)) = (int32_t)_y2;
            }

        } else if (_rem_part == 3) {

            _x0 = *((v4s*)(_p_in_part_iter + 0 * _inner_len_aligned));
            _x1 = *((v4s*)(_p_in_part_iter + 1 * _inner_len_aligned));
            _x2 = *((v4s*)(_p_in_part_iter + 2 * _inner_len_aligned));

            // compute _y0, _y1
            _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_1);
            _tmp2 = __builtin_shuffle(_x2, _zero, _SHUFFLE_MASK_1_1);

            // if (_rem_blk >= 1), always true
            _y0 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_1);
            *((int32_t*)(_p_out_part_iter + 0 * _outer_len_aligned)) = (int32_t)_y0;

            if (_rem_blk >= 2) {
                _y1 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_2);
                *((int32_t*)(_p_out_part_iter + 1 * _outer_len_aligned)) = (int32_t)_y1;
            }
            if (_rem_blk >= 3) {
                // compute _y2, _y3
                _tmp1 = __builtin_shuffle(_x0, _x1, _SHUFFLE_MASK_1_2);
                _tmp2 = __builtin_shuffle(_x2, _zero, _SHUFFLE_MASK_1_2);
                _y2 = __builtin_shuffle(_tmp1, _tmp2, _SHUFFLE_MASK_2_1);
                *((int32_t*)(_p_out_part_iter + 2 * _outer_len_aligned)) = (int32_t)_y2;
            }

        }
    }

}
