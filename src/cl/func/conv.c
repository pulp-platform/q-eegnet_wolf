/**
 * @file conv.h
 * @author Tibor Schneider
 * @date 2020/01/25
 * @brief Implementation of convolution in the valid region
 *
 * The source code was taken and modified from pulp-platform/pulp-dsp:
 * @see https://github.com/pulp-platform/pulp-dsp/blob/master/src/FilteringFunctions/kernels/plp_conv_i8s_xpulpv2.c
 *
 * What is changed? (measurement, convolution of 1188 x 64)
 * 0. Original method:                     (68579 Cycles, 63147 Instructions)
 * 1. Load a always aligned                (68027 Cycles, 67361 Instructions)
 * 2. Remaining elements only if necessary (65783 Cycles, 64242 Instructions)
 * 3. Only load one new element of a       (69723 Cycles, 68492 Instructions), but maybe better for parallel?
 */

/*
 * Copyright (C) 2020 ETH Zurich. All rights reserved.
 *
 * Author: Moritz Scherer, Tibor Schneider, ETH Zurich
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


#ifdef NO_SIMD
#define CONV_VERSION -1
#endif//NO_SIMD

#ifndef CONV_VERSION
#define CONV_VERSION 2
#endif//CONV_VERSION

#include "rt/rt_api.h"
#include "plp_math.h"
#include "functional.h"

#define shufflemask1 (v4s){3,2,1,0}

#if CONV_VERSION == 0

#define shufflemask2 (v4s){1,2,3,5}
#define shufflemask3 (v4s){2,3,5,6}

#else

#define shufflemask2 (v4s){1,2,3,4}
#define shufflemask3 (v4s){2,3,4,5}
#define shufflemask4 (v4s){3,4,5,6}

#endif

#if CONV_VERSION == -1

void func_conv(const int8_t* p_a,
               unsigned int a_len,
               const int8_t* p_b,
               unsigned int b_len,
               int32_t* p_res) {

    // Flip vectors a and b if b is larger than a
    if (a_len < b_len) {
        const int8_t* p_tmp = p_a;
        p_a = p_b;
        p_b = p_tmp;
        unsigned int tmp_len = a_len;
        a_len = b_len;
        b_len = tmp_len;
    }

    const int8_t* p_x;
    const int8_t* p_y;

    int res_len = a_len - b_len + 1;

    for (int i_out = 0; i_out < res_len; i_out++) {

        p_x = p_a + i_out;
        p_y = p_b + b_len - 1;

        int32_t acc = 0;

        for (int i_in = 0; i_in < b_len; i_in++) {
            acc += (*(p_x++)) * (*(p_y--));
        }

        p_res[i_out] = acc;

    }

}

void func_conv_scale(const int8_t* p_a,
                     unsigned int a_len,
                     const int8_t* p_b,
                     unsigned int b_len,
                     int32_t div_factor,
                     int32_t offset,
                     int8_t* p_res) {

    // Flip vectors a and b if b is larger than a
    if (a_len < b_len) {
        const int8_t* p_tmp = p_a;
        p_a = p_b;
        p_b = p_tmp;
        unsigned int tmp_len = a_len;
        a_len = b_len;
        b_len = tmp_len;
    }

    const int8_t* p_x;
    const int8_t* p_y;

    int res_len = a_len - b_len + 1;

    for (int i_out = 0; i_out < res_len; i_out++) {

        p_x = p_a + i_out;
        p_y = p_b + b_len - 1;

        int32_t acc = offset;

        for (int i_in = 0; i_in < b_len; i_in++) {
            acc += (*(p_x++)) * (*(p_y--));
        }

        acc = acc / div_factor;
        acc = __CLIP_R(acc, 127);

        p_res[i_out] = (int8_t)acc;

    }

}

#else //CONV_VERSION >= 0

void func_conv(const int8_t* p_a,
               unsigned int a_len,
               const int8_t* p_b,
               unsigned int b_len,
               int32_t* p_res) {

    // Flip vectors a and b if b is larger than a
    if (a_len < b_len) {
        const int8_t* p_tmp = p_a;
        p_a = p_b;
        p_b = p_tmp;
        unsigned int tmp_len = a_len;
        a_len = b_len;
        b_len = tmp_len;
    }

    const int8_t* px;                            /* Intermediate inputA pointer */
    const int8_t* py;                            /* Intermediate inputB pointer */
    const int8_t* p_b_tmp;                       /* Intermediate pointers */
    int32_t sum;                                 /* Accumulators */
    uint32_t block_size;                         /* Loop counters */
    uint32_t j, k, count, blk_cnt;               /* Loop counters */

    // for loop unroll
    int32_t acc0, acc1, acc2, acc3;              /* Accumulators */

    int32_t temp1, temp2;
    v4s xmask[] = {(v4s){0,0,0,0}, (v4s){0xff,0,0,0}, (v4s){0xff,0xff,0,0}, (v4s){0xff,0xff,0xff,0}};
    v4s ymask[] = {(v4s){0,0,0,0}, (v4s){0,0,0,0xff}, (v4s){0,0,0xff,0xff}, (v4s){0,0xff,0xff,0xff}};
    v4s mask;

    v4s _x1, _x2, _x3, _x4;
    v4s _y1, _y2;

#if CONV_VERSION >= 3
    v4s _x5;
#endif

    block_size = a_len - (b_len - 1U);

    /* --------------------------
     * Initializations of stage2
     * ------------------------*/

    /* sum = x[0] * y[b_len-1] + x[1] * y[b_len-2] +...+ x[b_len-1] * y[0]
     * sum = x[1] * y[b_len-1] + x[2] * y[b_len-2] +...+ x[b_len]   * y[0]
     * ....
     * sum = x[a_len-b_len-2] * y[b_len-1] + x[a_len] * y[b_len-2] +...+ x[a_len-1] * y[0]
     */

    /* Working pointer of inputA */
    px = p_a;

    /* Working pointer of inputB */
    p_b_tmp = p_b + (b_len - 1U);
    py = p_b_tmp;

    /* count is index by which the pointer p_a to be incremented */
    count = 0U;

    /* -------------------
     * Stage2 process
     * ------------------*/

    /* Stage2 depends on b_len as in this stage b_len number of MACS are performed.
     * So, to loop unroll over block_size,
     * b_len should be greater than or equal to 4 */
    if (b_len >= 4U) {

        /* Loop unrolling: Compute 4 outputs at a time */
        blk_cnt = block_size >> 2U;
        while (blk_cnt > 0U) {
            /* Set all accumulators to zero */
            acc0 = 0;
            acc1 = 0;
            acc2 = 0;
            acc3 = 0;

            /* Apply loop unrolling and compute 4 MACs simultaneously. */
            k = b_len >> 2U;

#if CONV_VERSION == 0

            /* First part of the processing with loop unrolling.  Compute 4 MACs at a
             * a second loop below computes MACs for the remaining 1 to 3 samples. */
            do {
                /* Read y[b_len - 1] sample */
                _x1 = *((v4s*)px); // {x[0],x[1],x[2],x[3]}
                _x4 = *((v4s*)(px+3)); // {x[3],x[4],x[5],x[6]}
                _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

                px+=4U;
                py-=4U;

                _x2 = __builtin_shuffle(_x1,_x4, shufflemask2); // {x[1],x[2],x[3],x[4]}
                _x3 = __builtin_shuffle(_x1,_x4, shufflemask3); // {x[2],x[3],x[4],x[5]}

                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1); // {y[b_len - 1],y[b_len - 2],y[b_len - 3],y[b_len - 4]} 

                acc0 = __SUMDOTP4(_x1,_y1,acc0);
                acc1 = __SUMDOTP4(_x2,_y1,acc1);
                acc2 = __SUMDOTP4(_x3,_y1,acc2);
                acc3 = __SUMDOTP4(_x4,_y1,acc3);

            } while (--k);

            /* If the b_len is not a multiple of 4, compute any remaining MACs here.
            ** No loop unrolling is used. */

            k = b_len % 0x4U;

            _x1 = *((v4s*)px); // {x[0],x[1],x[2],x[3]}
            _x4 = *((v4s*)(px+3)); // {x[3],x[4],x[5],x[6]}
            _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

            mask = ymask[k];

            _x2 = __builtin_shuffle(_x1,_x4, shufflemask2); // {x[1],x[2],x[3],x[4]}
            _x3 = __builtin_shuffle(_x1,_x4, shufflemask3); // {x[2],x[3],x[4],x[5]}

            _y1 = __AND4(_y1,mask);	  
            _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);

            /* Perform the multiply-accumulate */

            acc0 = __SUMDOTP4(_x1,_y1,acc0);
            acc1 = __SUMDOTP4(_x2,_y1,acc1);
            acc2 = __SUMDOTP4(_x3,_y1,acc2);
            acc3 = __SUMDOTP4(_x4,_y1,acc3);

#elif CONV_VERSION == 1 || CONV_VERSION == 2

            /* First part of the processing with loop unrolling.  Compute 4 MACs at a
             * a second loop below computes MACs for the remaining 1 to 3 samples. */
            do {
                /* Read y[b_len - 1] sample */
                _x1 = *((v4s*)px); // {x[0],x[1],x[2],x[3]}
                _x4 = *((v4s*)(px+4)); // {x[4],x[5],x[6],x[7]}
                _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

                px+=4U;
                py-=4U;

                _x2 = __builtin_shuffle(_x1,_x4, shufflemask2); // {x[1],x[2],x[3],x[4]}
                _x3 = __builtin_shuffle(_x1,_x4, shufflemask3); // {x[2],x[3],x[4],x[5]}
                _x4 = __builtin_shuffle(_x1,_x4, shufflemask4); // {x[2],x[3],x[4],x[5]}

                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1); // {y[b_len - 1],y[b_len - 2],y[b_len - 3],y[b_len - 4]} 

                acc0 = __SUMDOTP4(_x1,_y1,acc0);
                acc1 = __SUMDOTP4(_x2,_y1,acc1);
                acc2 = __SUMDOTP4(_x3,_y1,acc2);
                acc3 = __SUMDOTP4(_x4,_y1,acc3);

            } while (--k);

            /* If the b_len is not a multiple of 4, compute any remaining MACs here.
            ** No loop unrolling is used. */

            k = b_len % 0x4U;

#if CONV_VERSION == 2
            if (k > 0) {
#endif
                _x1 = *((v4s*)px); // {x[0],x[1],x[2],x[3]}
                _x4 = *((v4s*)(px+4)); // {x[4],x[5],x[6],x[7]}
                _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

                mask = ymask[k];

                _x2 = __builtin_shuffle(_x1,_x4, shufflemask2); // {x[1],x[2],x[3],x[4]}
                _x3 = __builtin_shuffle(_x1,_x4, shufflemask3); // {x[2],x[3],x[4],x[5]}
                _x4 = __builtin_shuffle(_x1,_x4, shufflemask4); // {x[3],x[4],x[5],x[6]}

                _y1 = __AND4(_y1,mask);	  
                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);

                /* Perform the multiply-accumulate */

                acc0 = __SUMDOTP4(_x1,_y1,acc0);
                acc1 = __SUMDOTP4(_x2,_y1,acc1);
                acc2 = __SUMDOTP4(_x3,_y1,acc2);
                acc3 = __SUMDOTP4(_x4,_y1,acc3);

#if CONV_VERSION == 2
            }
#endif

#elif CONV_VERSION == 3

            /* First part of the processing with loop unrolling.  Compute 4 MACs at a
             * a second loop below computes MACs for the remaining 1 to 3 samples. */
            // prepare the next load
            _x5 = *((v4s*)px);
            do {
                /* Read y[b_len - 1] sample */
                _x1 = _x5;             // {x[0],x[1],x[2],x[3]}, loaded from before
                _x5 = *((v4s*)(px+4)); // {x[4],x[5],x[6],x[7]}
                _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

                px+=4U;
                py-=4U;

                _x2 = __builtin_shuffle(_x1,_x5, shufflemask2); // {x[1],x[2],x[3],x[4]}
                _x3 = __builtin_shuffle(_x1,_x5, shufflemask3); // {x[2],x[3],x[4],x[5]}
                _x4 = __builtin_shuffle(_x1,_x5, shufflemask4); // {x[2],x[3],x[4],x[5]}

                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1); // {y[b_len - 1],y[b_len - 2],y[b_len - 3],y[b_len - 4]} 

                acc0 = __SUMDOTP4(_x1,_y1,acc0);
                acc1 = __SUMDOTP4(_x2,_y1,acc1);
                acc2 = __SUMDOTP4(_x3,_y1,acc2);
                acc3 = __SUMDOTP4(_x4,_y1,acc3);

            } while (--k);

            /* If the b_len is not a multiple of 4, compute any remaining MACs here.
            ** No loop unrolling is used. */

            k = b_len % 0x4U;

            if (k > 0) {

                _x1 = _x5;             // {x[0],x[1],x[2],x[3]}, loaded from before
                _x5 = *((v4s*)(px+4)); // {x[4],x[5],x[6],x[7]}
                _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

                mask = ymask[k];

                _x2 = __builtin_shuffle(_x1,_x5, shufflemask2); // {x[1],x[2],x[3],x[4]}
                _x3 = __builtin_shuffle(_x1,_x5, shufflemask3); // {x[2],x[3],x[4],x[5]}
                _x4 = __builtin_shuffle(_x1,_x5, shufflemask4); // {x[3],x[4],x[5],x[6]}

                _y1 = __AND4(_y1,mask);	  
                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);

                /* Perform the multiply-accumulate */

                acc0 = __SUMDOTP4(_x1,_y1,acc0);
                acc1 = __SUMDOTP4(_x2,_y1,acc1);
                acc2 = __SUMDOTP4(_x3,_y1,acc2);
                acc3 = __SUMDOTP4(_x4,_y1,acc3);

            }

#endif //CONV_VERSION

            /* Store the result in the accumulator in the destination buffer. */
            *p_res++ = acc0;
            *p_res++ = acc1;
            *p_res++ = acc2;
            *p_res++ = acc3;

            /* Increment the pointer p_a index, count by 4 */
            count += 4U;

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = p_a + count;
            py = p_b_tmp;

            /* Decrement the loop counter */
            blk_cnt--;
        }

        /* If the block_size is not a multiple of 4, compute any remaining output samples here.
        ** No loop unrolling is used. */
        blk_cnt = block_size % 0x4U;

        while (blk_cnt > 0U) {
            /* Accumulator is made zero for every iteration */

            _y1 = *((v4s*)(py-3));
            _x1 = *((v4s*)(px));
            sum = 0;
            _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);

            /* Loop unrolling: Compute 8 outputs at a time */
            k = b_len >> 2U;
            while (k > 0U) { 
                sum = __SUMDOTP4(_x1,_y1,sum);

                _y1 = *((v4s*)(py-7));
                _x1 = *((v4s*)(px+4));

                px += 4U;
                py -= 4U;

                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);
                k--;
            }

            /* Loop unrolling: Compute remaining outputs */
            k = b_len % 0x4U;

            mask = xmask[k];
            _x1 = __AND4(_x1,mask);
            sum = __SUMDOTP4(_x1,_y1,sum);

            /* Store the result in the accumulator in the destination buffer. */
            *p_res++ = sum;

            /* Increment the MAC count */
            count++;

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = p_a + count;
            py = p_b_tmp;

            /* Decrement the loop counter */
            blk_cnt--;
        }
    }
    else {
        /* If the b_len is not a multiple of 4,
         * the block_size loop cannot be unrolled by 4 */
        blk_cnt = block_size;

        while (blk_cnt > 0U) {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            /* b_len number of MACS should be performed */
            k = b_len;
            mask = xmask[k];

            _y1 = *((v4s*)(py-3));
            _x1 = *((v4s*)(px));

            _x1 = __AND4(_x1,mask);
            _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);
	  
            sum = __SUMDOTP4(_x1,_y1,sum);
      
            /* Store the result in the accumulator in the destination buffer. */
            *p_res++ = sum;

            /* Increment the MAC count */
            count++;

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = p_a + count;
            py = p_b_tmp;

            /* Decrement the loop counter */
            blk_cnt--;
        }
    }

}


/**
 * @brief Compute the convolution of vectors a and b and scales the result back to 8 bit
 *
 * The operation is performed only in the valid range. This means that the output
 * size is a_len - b_len + 1.
 *
 * The source code was taken and modified from pulp-platform/pulp-dsp:
 * @see https://github.com/pulp-platform/pulp-dsp/blob/master/src/FilteringFunctions/kernels/plp_conv_i8s_xpulpv2.c
 *
 * @warning Data must be already present in L1 memory, and the output vector must 
 * be allocated
 *
 * @warning the smaller vector must be at least of length 4
 *
 * @param p_a Pointer to vector a on L1 memory
 * @param a_len Length of vector a, a_len >= 2
 * @param p_b Pointer to vector b on L1 memory
 * @param b_len Length of vector b, b_len >= 2
 * @param div_factor factor by which the result is divided
 * @param offset Bias which is added to the result before division.
 * @param p_res Pointer to the output vector.
 */
void func_conv_scale(const int8_t* p_a,
                     unsigned int a_len,
                     const int8_t* p_b,
                     unsigned int b_len,
                     int32_t div_factor,
                     int32_t offset,
                     int8_t* p_res) {

    // Flip vectors a and b if b is larger than a
    if (a_len < b_len) {
        const int8_t* p_tmp = p_a;
        p_a = p_b;
        p_b = p_tmp;
        unsigned int tmp_len = a_len;
        a_len = b_len;
        b_len = tmp_len;
    }

    const int8_t* px;                            /* Intermediate inputA pointer */
    const int8_t* py;                            /* Intermediate inputB pointer */
    const int8_t* p_b_tmp;                       /* Intermediate pointers */
    int32_t sum;                                 /* Accumulators */
    uint32_t block_size;                         /* Loop counters */
    uint32_t j, k, count, blk_cnt;               /* Loop counters */

    // for loop unroll
    int32_t acc0, acc1, acc2, acc3;              /* Accumulators */

    int32_t temp1, temp2;
    v4s xmask[] = {(v4s){0,0,0,0}, (v4s){0xff,0,0,0}, (v4s){0xff,0xff,0,0}, (v4s){0xff,0xff,0xff,0}};
    v4s ymask[] = {(v4s){0,0,0,0}, (v4s){0,0,0,0xff}, (v4s){0,0,0xff,0xff}, (v4s){0,0xff,0xff,0xff}};
    v4s mask;

    v4s _x1, _x2, _x3, _x4;
    v4s _y1, _y2;

#if CONV_VERSION >= 3
    v4s _x5;
#endif

    block_size = a_len - (b_len - 1U);

    // this array stores the temporary result of the remaining elements to be transformed afterwards
    int32_t tmp_res[3] = {0, 0, 0};
    int32_t* p_tmp_res_iter = tmp_res; // counter to the current element of p_tmp_res_iter

    /* --------------------------
     * Initializations of stage2
     * ------------------------*/

    /* sum = x[0] * y[b_len-1] + x[1] * y[b_len-2] +...+ x[b_len-1] * y[0]
     * sum = x[1] * y[b_len-1] + x[2] * y[b_len-2] +...+ x[b_len]   * y[0]
     * ....
     * sum = x[a_len-b_len-2] * y[b_len-1] + x[a_len] * y[b_len-2] +...+ x[a_len-1] * y[0]
     */

    /* Working pointer of inputA */
    px = p_a;

    /* Working pointer of inputB */
    p_b_tmp = p_b + (b_len - 1U);
    py = p_b_tmp;

    /* count is index by which the pointer p_a to be incremented */
    count = 0U;

    /* -------------------
     * Stage2 process
     * ------------------*/

    /* Stage2 depends on b_len as in this stage b_len number of MACS are performed.
     * So, to loop unroll over block_size,
     * b_len should be greater than or equal to 4 */
    if (b_len >= 4U) {

        /* Loop unrolling: Compute 4 outputs at a time */
        blk_cnt = block_size >> 2U;
        while (blk_cnt > 0U) {
            /* Set all accumulators to zero */
            acc0 = 0;
            acc1 = 0;
            acc2 = 0;
            acc3 = 0;

            /* Apply loop unrolling and compute 4 MACs simultaneously. */
            k = b_len >> 2U;

#if CONV_VERSION == 0

            /* First part of the processing with loop unrolling.  Compute 4 MACs at a
             * a second loop below computes MACs for the remaining 1 to 3 samples. */
            do {
                /* Read y[b_len - 1] sample */
                _x1 = *((v4s*)px); // {x[0],x[1],x[2],x[3]}
                _x4 = *((v4s*)(px+3)); // {x[3],x[4],x[5],x[6]}
                _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

                px+=4U;
                py-=4U;

                _x2 = __builtin_shuffle(_x1,_x4, shufflemask2); // {x[1],x[2],x[3],x[4]}
                _x3 = __builtin_shuffle(_x1,_x4, shufflemask3); // {x[2],x[3],x[4],x[5]}

                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1); // {y[b_len - 1],y[b_len - 2],y[b_len - 3],y[b_len - 4]} 

                acc0 = __SUMDOTP4(_x1,_y1,acc0);
                acc1 = __SUMDOTP4(_x2,_y1,acc1);
                acc2 = __SUMDOTP4(_x3,_y1,acc2);
                acc3 = __SUMDOTP4(_x4,_y1,acc3);

            } while (--k);

            /* If the b_len is not a multiple of 4, compute any remaining MACs here.
            ** No loop unrolling is used. */

            k = b_len % 0x4U;

            _x1 = *((v4s*)px); // {x[0],x[1],x[2],x[3]}
            _x4 = *((v4s*)(px+3)); // {x[3],x[4],x[5],x[6]}
            _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

            mask = ymask[k];

            _x2 = __builtin_shuffle(_x1,_x4, shufflemask2); // {x[1],x[2],x[3],x[4]}
            _x3 = __builtin_shuffle(_x1,_x4, shufflemask3); // {x[2],x[3],x[4],x[5]}

            _y1 = __AND4(_y1,mask);	  
            _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);

            /* Perform the multiply-accumulate */

            acc0 = __SUMDOTP4(_x1,_y1,acc0);
            acc1 = __SUMDOTP4(_x2,_y1,acc1);
            acc2 = __SUMDOTP4(_x3,_y1,acc2);
            acc3 = __SUMDOTP4(_x4,_y1,acc3);

#elif CONV_VERSION == 1 || CONV_VERSION == 2

            /* First part of the processing with loop unrolling.  Compute 4 MACs at a
             * a second loop below computes MACs for the remaining 1 to 3 samples. */
            do {
                /* Read y[b_len - 1] sample */
                _x1 = *((v4s*)px); // {x[0],x[1],x[2],x[3]}
                _x4 = *((v4s*)(px+4)); // {x[4],x[5],x[6],x[7]}
                _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

                px+=4U;
                py-=4U;

                _x2 = __builtin_shuffle(_x1,_x4, shufflemask2); // {x[1],x[2],x[3],x[4]}
                _x3 = __builtin_shuffle(_x1,_x4, shufflemask3); // {x[2],x[3],x[4],x[5]}
                _x4 = __builtin_shuffle(_x1,_x4, shufflemask4); // {x[2],x[3],x[4],x[5]}

                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1); // {y[b_len - 1],y[b_len - 2],y[b_len - 3],y[b_len - 4]} 

                acc0 = __SUMDOTP4(_x1,_y1,acc0);
                acc1 = __SUMDOTP4(_x2,_y1,acc1);
                acc2 = __SUMDOTP4(_x3,_y1,acc2);
                acc3 = __SUMDOTP4(_x4,_y1,acc3);

            } while (--k);

            /* If the b_len is not a multiple of 4, compute any remaining MACs here.
            ** No loop unrolling is used. */

            k = b_len % 0x4U;

#if CONV_VERSION == 2
            if (k > 0) {
#endif
                _x1 = *((v4s*)px); // {x[0],x[1],x[2],x[3]}
                _x4 = *((v4s*)(px+4)); // {x[4],x[5],x[6],x[7]}
                _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

                mask = ymask[k];

                _x2 = __builtin_shuffle(_x1,_x4, shufflemask2); // {x[1],x[2],x[3],x[4]}
                _x3 = __builtin_shuffle(_x1,_x4, shufflemask3); // {x[2],x[3],x[4],x[5]}
                _x4 = __builtin_shuffle(_x1,_x4, shufflemask4); // {x[3],x[4],x[5],x[6]}

                _y1 = __AND4(_y1,mask);	  
                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);

                /* Perform the multiply-accumulate */

                acc0 = __SUMDOTP4(_x1,_y1,acc0);
                acc1 = __SUMDOTP4(_x2,_y1,acc1);
                acc2 = __SUMDOTP4(_x3,_y1,acc2);
                acc3 = __SUMDOTP4(_x4,_y1,acc3);

#if CONV_VERSION == 2
            }
#endif

#elif CONV_VERSION == 3

            /* First part of the processing with loop unrolling.  Compute 4 MACs at a
             * a second loop below computes MACs for the remaining 1 to 3 samples. */
            // prepare the next load
            _x5 = *((v4s*)px);
            do {
                /* Read y[b_len - 1] sample */
                _x1 = _x5;             // {x[0],x[1],x[2],x[3]}, loaded from before
                _x5 = *((v4s*)(px+4)); // {x[4],x[5],x[6],x[7]}
                _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

                px+=4U;
                py-=4U;

                _x2 = __builtin_shuffle(_x1,_x5, shufflemask2); // {x[1],x[2],x[3],x[4]}
                _x3 = __builtin_shuffle(_x1,_x5, shufflemask3); // {x[2],x[3],x[4],x[5]}
                _x4 = __builtin_shuffle(_x1,_x5, shufflemask4); // {x[2],x[3],x[4],x[5]}

                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1); // {y[b_len - 1],y[b_len - 2],y[b_len - 3],y[b_len - 4]} 

                acc0 = __SUMDOTP4(_x1,_y1,acc0);
                acc1 = __SUMDOTP4(_x2,_y1,acc1);
                acc2 = __SUMDOTP4(_x3,_y1,acc2);
                acc3 = __SUMDOTP4(_x4,_y1,acc3);

            } while (--k);

            /* If the b_len is not a multiple of 4, compute any remaining MACs here.
            ** No loop unrolling is used. */

            k = b_len % 0x4U;

            if (k > 0) {

                _x1 = _x5;             // {x[0],x[1],x[2],x[3]}, loaded from before
                _x5 = *((v4s*)(px+4)); // {x[4],x[5],x[6],x[7]}
                _y1 = *((v4s*)(py-3)); // {y[b_len - 4],y[b_len - 3],y[b_len - 2],y[b_len - 1]} 

                mask = ymask[k];

                _x2 = __builtin_shuffle(_x1,_x5, shufflemask2); // {x[1],x[2],x[3],x[4]}
                _x3 = __builtin_shuffle(_x1,_x5, shufflemask3); // {x[2],x[3],x[4],x[5]}
                _x4 = __builtin_shuffle(_x1,_x5, shufflemask4); // {x[3],x[4],x[5],x[6]}

                _y1 = __AND4(_y1,mask);	  
                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);

                /* Perform the multiply-accumulate */

                acc0 = __SUMDOTP4(_x1,_y1,acc0);
                acc1 = __SUMDOTP4(_x2,_y1,acc1);
                acc2 = __SUMDOTP4(_x3,_y1,acc2);
                acc3 = __SUMDOTP4(_x4,_y1,acc3);

            }

#endif //CONV_VERSION

            // scale the result and store it in the destination buffer
            *((int32_t*)p_res) = (int32_t)func_transform_32to8_bias_elem(acc0, acc1, acc2, acc3, div_factor, offset);
            p_res += 4;

            /* Increment the pointer p_a index, count by 4 */
            count += 4U;

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = p_a + count;
            py = p_b_tmp;

            /* Decrement the loop counter */
            blk_cnt--;
        }

        /* If the block_size is not a multiple of 4, compute any remaining output samples here.
        ** No loop unrolling is used. */
        blk_cnt = block_size % 0x4U;

        if (blk_cnt > 0) {
            while (blk_cnt > 0U) {
                /* Accumulator is made zero for every iteration */

                _y1 = *((v4s*)(py-3));
                _x1 = *((v4s*)(px));
                sum = 0;
                _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);

                /* Loop unrolling: Compute 8 outputs at a time */
                k = b_len >> 2U;
                while (k > 0U) { 
                    sum = __SUMDOTP4(_x1,_y1,sum);

                    _y1 = *((v4s*)(py-7));
                    _x1 = *((v4s*)(px+4));

                    px += 4U;
                    py -= 4U;

                    _y1 = __builtin_shuffle(_y1,_y1,shufflemask1);
                    k--;
                }

                /* Loop unrolling: Compute remaining outputs */
                k = b_len % 0x4U;

                mask = xmask[k];
                _x1 = __AND4(_x1,mask);
                sum = __SUMDOTP4(_x1,_y1,sum);

                /* Store the result in the accumulator in the destination buffer. */
                *p_tmp_res_iter++ = sum;

                /* Increment the MAC count */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = p_a + count;
                py = p_b_tmp;

                /* Decrement the loop counter */
                blk_cnt--;
            }

            // transform the values and store the result
            acc0 = tmp_res[0];
            acc1 = tmp_res[1];
            acc2 = tmp_res[2];
            *((int32_t*)p_res) = (int32_t)func_transform_32to8_bias_elem(acc0, acc1, acc2, 0, div_factor, offset);

        }
    }
    else {
        printf("func_conv_scale: Error: smaller vector must be at least 4 elements long");
    }

}

#endif //CONV_VERSION == -1
