/**
 * @file conv.h
 * @author Tibor Schneider
 * @date 2020/01/25
 * @brief Implementation of transformation from 32bit to 8bit
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
#include "stdio.h"

/**
 * @brief Convert a vector of 32bits back to 8bit (by scaling and shifting)
 *
 * Per element k, p_res[k] = (p_in[k * stride] + div_factor / 2) / div_factor
 * 
 * @warning Data must be already present in L1 memory, and the output vector must 
 * be allocated
 *
 * @param p_in Pointer to input vector a on L1 memory
 * @param len Length of the input and output vector
 * @param div_factor factor by which to divide
 * @param stride Collect elements from p_in with distance stride apart. Set to 1 for default stride
 * @param p_res Pointer to the output vector.
 */
void func_transform_32to8(const int32_t* p_in,
                          unsigned int len,
                          int32_t div_factor,
                          unsigned int stride,
                          int8_t* p_res) {

    int32_t _a, _b, _c, _d;          // temporary values
    const int32_t* _p_x = p_in;      // pointer to current element in x

    unsigned int _num_blk = len / 4;
    unsigned int _num_rem = len % 4;

    // do the elements which can be unrolled
    while (_num_blk > 0) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);
        _c = *(_p_x + 2 * stride);
        _d = *(_p_x + 3 * stride);

        _p_x += 4 * stride;

        _a = _a / div_factor;
        _b = _b / div_factor;
        _c = _c / div_factor;
        _d = _d / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);
        _c = __CLIP_R(_c, 127);
        _d = __CLIP_R(_d, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, _c, _d);
        
        p_res += 4;
        _num_blk--;
    }

    if (_num_rem == 1) {

        _a = *_p_x;
        _a = _a / div_factor;
        _a = __CLIP_R(_a, 127);
        *((int32_t*)p_res) = (int32_t)__PACK4(_a, 0, 0, 0);

    } else if (_num_rem == 2) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);

        _a = _a / div_factor;
        _b = _b / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, 0, 0);

    } else if (_num_rem == 3) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);
        _c = *(_p_x + 2 * stride);

        _a = _a / div_factor;
        _b = _b / div_factor;
        _c = _c / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);
        _c = __CLIP_R(_c, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, _c, 0);
    }
}

/**
 * @brief Convert a vector of 32bits back to 8bit (by scaling and shifting)
 *
 * Per element k, p_res[k] = (p_in[k * stride] + bias + div_factor / 2) / div_factor
 * 
 * @warning Data must be already present in L1 memory, and the output vector must 
 * be allocated
 *
 * @param p_in Pointer to input vector a on L1 memory
 * @param len Length of the input and output vector
 * @param div_factor factor by which to divide
 * @param bias offset added before dividing by the factor
 * @param stride Collect elements from p_in with distance stride apart. Set to 1 for default stride
 * @param p_res Pointer to the output vector.
 */
void func_transform_32to8_bias(const int32_t* p_in,
                                 unsigned int len,
                                 int32_t div_factor,
                                 int32_t bias,
                                 unsigned int stride,
                                 int8_t* p_res) {

    int32_t _a, _b, _c, _d;          // temporary values
    const int32_t* _p_x = p_in;      // pointer to current element in x

    unsigned int _num_blk = len / 4;
    unsigned int _num_rem = len % 4;

    // do the elements which can be unrolled
    while (_num_blk > 0) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);
        _c = *(_p_x + 2 * stride);
        _d = *(_p_x + 3 * stride);

        _a += bias;
        _b += bias;
        _c += bias;
        _d += bias;

        _a = _a / div_factor;
        _b = _b / div_factor;
        _c = _c / div_factor;
        _d = _d / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);
        _c = __CLIP_R(_c, 127);
        _d = __CLIP_R(_d, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, _c, _d);
        
        p_res += 4;
        _p_x += 4 * stride;
        _num_blk--;
    }

    if (_num_rem == 1) {

        _a = *_p_x;
        _a += bias;
        _a = _a / div_factor;
        _a = __CLIP_R(_a, 127);
        *((int32_t*)p_res) = (int32_t)__PACK4(_a, 0, 0, 0);

    } else if (_num_rem == 2) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);

        _a += bias;
        _b += bias;

        _a = _a / div_factor;
        _b = _b / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, 0, 0);

    } else if (_num_rem == 3) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);
        _c = *(_p_x + 2 * stride);

        _a += bias;
        _b += bias;
        _c += bias;

        _a = _a / div_factor;
        _b = _b / div_factor;
        _c = _c / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);
        _c = __CLIP_R(_c, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, _c, 0);
    }

}
