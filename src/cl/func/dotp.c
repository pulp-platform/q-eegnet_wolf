/**
 * @file functional.h
 * @author Tibor Schneider
 * @date 2020/01/31
 * @brief this file contains the functions for dot product.
 *
 * This function was copied and modified from PULP-DSP
 * @see https://github.com/pulp-platform/pulp-dsp/blob/master/src/BasicMathFunctions/kernels/plp_dot_prod_i8v_xpulpv2.c
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

#ifdef NO_SIMD

/**
 * @brief computes dot product of the two vectors p_a and p_b
 *
 * @param p_a Pointer to first vector on L1 memory, should be aligned
 * @param p_b Pointer to second vector on L1 memory, should be aligned
 * @param length Lenght (number of elements) of both vectors
 * @return dot product
 */
int32_t func_dotp(const int8_t* p_a,
                  const int8_t* p_b,
                  unsigned int length) {

    //return func_dotp_slow(p_a, 1, p_b, 1, length);
    int32_t acc = 0;
    for (int i = 0; i < length; i++) {
        acc += p_a[i] * p_b[i];
    }
    return acc;

}

int32_t func_dotp_slow(const int8_t* p_a,
                       unsigned int a_stride,
                       const int8_t* p_b,
                       unsigned int b_stride,
                       unsigned int length) {

    int32_t acc = 0;
    for (int i = 0; i < length; i++) {
        acc += p_a[i * a_stride] * p_b[i * b_stride];
    }
    return acc;

}

#else//NO_SIMD

/**
 * @brief computes dot product of the two vectors p_a and p_b
 *
 * @param p_a Pointer to first vector on L1 memory, should be aligned
 * @param p_b Pointer to second vector on L1 memory, should be aligned
 * @param length Lenght (number of elements) of both vectors
 * @return dot product
 */
int32_t func_dotp(const int8_t* p_a,
                  const int8_t* p_b,
                  unsigned int length) {

    unsigned int _num_blk = length / 8;
    unsigned int _rem_blk = length % 8;

    const v4s* _p_a_iter = (v4s*)p_a;
    const v4s* _p_b_iter = (v4s*)p_b;

    int32_t _acc0 = 0, _acc1 = 0;
    v4s _a0, _a1, _b0, _b1;

    v4s _rem_mask = (v4s){0, 5, 6, 7};

    // do the main bulk of the computation

    while (_num_blk > 0) {

        _a0 = *_p_a_iter++;
        _b0 = *_p_b_iter++;
        _a1 = *_p_a_iter++;
        _b1 = *_p_b_iter++;

        _acc0 = __SUMDOTP4(_a0, _b0, _acc0);
        _acc1 = __SUMDOTP4(_a1, _b1, _acc1);

        _num_blk--;

    }

    // special case for _rem_blk = 4
    if (_rem_blk == 4) {
        _a0 = *_p_a_iter;
        _b0 = *_p_b_iter;
        _acc0 = __SUMDOTP4(_a0, _b0, _acc0);
        return _acc0 + _acc1;
    }

    // all other cases
    else if (_rem_blk > 0){

        switch (_rem_blk % 4) {
        case 2:
            _rem_mask = (v4s){0, 1, 6, 7};
            break;
        case 3:
            _rem_mask = (v4s){0, 1, 2, 7};
            break;
        }

        // For the remaining elements, mask the used blocks
        if (_rem_blk <= 4) {

            _a0 = *_p_a_iter;
            _b0 = *_p_b_iter;

            // mask _a0 such that it contains 0s at positions not used
            _a0 = __builtin_shuffle(_a0, (v4s){0, 0, 0, 0}, _rem_mask);

            _acc0 = __SUMDOTP4(_a0, _b0, _acc0);

        } else {

            _a0 = *_p_a_iter++;
            _b0 = *_p_b_iter++;
            _a1 = *_p_a_iter;
            _b1 = *_p_b_iter;

            // mask _a1 such that it contains 0s at positions not used
            _a1 = __builtin_shuffle(_a1, (v4s){0, 0, 0, 0}, _rem_mask);

            _acc0 = __SUMDOTP4(_a0, _b0, _acc0);
            _acc1 = __SUMDOTP4(_a1, _b1, _acc1);
        }
    }

    return _acc0 + _acc1;
}

/**
 * @brief computes dot product of the two vectors p_a and p_b without SIMD
 *
 * @param p_a Pointer to first vector on L1 memory, alignment does not matter at all
 * @param a_stride Distance between each element in the first vector.
 * @param p_b Pointer to second vector on L1 memory, alignment does not matter at all
 * @param b_stride Distance between each element in the second vector.
 * @param length Lenght (number of elements) of both vectors
 * @return dot product
 */
int32_t func_dotp_slow(const int8_t* p_a,
                       unsigned int a_stride,
                       const int8_t* p_b,
                       unsigned int b_stride,
                       unsigned int length) {

    const int8_t* _p_a_iter = p_a;
    const int8_t* _p_b_iter = p_b;

    int32_t _acc0 = 0, _acc1 = 0;
    int8_t _a0, _a1, _b0, _b1;

    unsigned int _num_blk = length / 2;
    unsigned int _num_rem = length % 2;

    while (_num_blk > 0) {

        _a0 = *_p_a_iter;
        _b0 = *_p_b_iter;
        _a1 = *(_p_a_iter + a_stride);
        _b1 = *(_p_b_iter + b_stride);

        _acc0 = __MAC(_acc0, _a0, _b0);
        _acc1 = __MAC(_acc1, _a1, _b1);

        // go to next element
        _p_a_iter += 2 * a_stride;
        _p_b_iter += 2 * b_stride;
        _num_blk--;

    }

    if (_num_rem) {
        _a0 = *_p_a_iter;
        _b0 = *_p_b_iter;

        _acc0 = __MAC(_acc0, _a0, _b0);
    }

    return _acc0 + _acc1;

}

#endif//NO_SIMD
