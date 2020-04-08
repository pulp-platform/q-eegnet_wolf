/**
 * @file functional.h
 * @author Tibor Schneider
 * @date 2020/01/30
 * @brief This file contains the definitions for all main mathematical functions
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


#ifndef __CL_FUNC_FUNCTIONAL_H__
#define __CL_FUNC_FUNCTIONAL_H__

/**
 * @brief Compute the convolution of vectors a and b
 *
 * The operation is performed only in the valid range. This means that the output
 * size is a_len - b_len + 1.
 *
 * The source code was taken and modified from pulp-platform/pulp-dsp:
 * @see https://github.com/pulp-platform/pulp-dsp/blob/master/src/FilteringFunctions/kernels/plp_conv_i8s_xpulpv2.c
 *
 *
 * @warning Data must be already present in L1 memory, and the output vector must 
 * be allocated
 *
 * @param p_a Pointer to vector a on L1 memory
 * @param a_len Length of vector a, a_len >= 2
 * @param p_b Pointer to vector b on L1 memory
 * @param b_len Length of vector b, b_len >= 2
 * @param p_res Pointer to the output vector.
 */
void func_conv(const int8_t* p_a,
               unsigned int a_len,
               const int8_t* p_b,
               unsigned int b_len,
               int32_t* p_res);

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
                     int8_t* p_res);

/**
 * @brief Compute the cross correlation of vectors a and b
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
 * @param p_a Pointer to vector a on L1 memory
 * @param a_len Length of vector a, a_len >= 2
 * @param p_b Pointer to vector b on L1 memory
 * @param b_len Length of vector b, b_len >= 2
 * @param p_res Pointer to the output vector.
 */
void func_xcorr(const int8_t* p_a,
                unsigned int a_len,
                const int8_t* p_b,
                unsigned int b_len,
                int32_t* p_res);

/**
 * @brief Compute the cross correlation of vectors a and b, and scales the result back to 8bit
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
 * @param p_a Pointer to vector a on L1 memory
 * @param a_len Length of vector a, a_len >= 2
 * @param p_b Pointer to vector b on L1 memory
 * @param b_len Length of vector b, b_len >= 2
 * @param div_factor factor by which the result is divided
 * @param offset Bias which is added to the result before division.
 * @param p_res Pointer to the output vector.
 */
void func_xcorr_scale(const int8_t* p_a,
                      unsigned int a_len,
                      const int8_t* p_b,
                      unsigned int b_len,
                      int32_t div_factor,
                      int32_t offset,
                      int8_t* p_res);

/**
 * @brief Convert a vector of 32bits back to 8bit (by scaling)
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
                          int8_t* p_res);

/**
 * @brief Convert 4 32bit integers back to 8 bits (by scaling)
 *
 * Per element k, y[k] = x[k] / div_factor
 *
 * @param x1 first element
 * @param x2 second element
 * @param x3 third element
 * @param x4 forth element
 * @param div_factor division factor
 * @return packed result
 */
inline v4s func_transform_32to8_elem(int32_t x1,
                                     int32_t x2,
                                     int32_t x3,
                                     int32_t x4,
                                     int32_t div_factor) {

    x1 = x1 / div_factor;
    x2 = x2 / div_factor;
    x3 = x3 / div_factor;
    x4 = x4 / div_factor;

    x1 = __CLIP_R(x1, 127);
    x2 = __CLIP_R(x2, 127);
    x3 = __CLIP_R(x3, 127);
    x4 = __CLIP_R(x4, 127);

    return __PACK4(x1, x2, x3, x4);
}

/**
 * @brief Convert a vector of 32bits back to 8bit (by scaling and shifting)
 *
 * Per element k, p_res[k] = (p_in[k * stride] + offset + div_factor / 2) / div_factor
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
                               int8_t* p_res);

/**
 * @brief Convert 4 32bit integers back to 8 bits (by scaling)
 *
 * Per element k, y[k] = (x[k] + offset) / div_factor
 *
 * @param x1 first element
 * @param x2 second element
 * @param x3 third element
 * @param x4 forth element
 * @param div_factor division factor
 * @return packed result
 */
inline v4s func_transform_32to8_bias_elem(int32_t x1,
                                          int32_t x2,
                                          int32_t x3,
                                          int32_t x4,
                                          int32_t div_factor,
                                          int32_t bias) {

    x1 = (x1 + bias) / div_factor;
    x2 = (x2 + bias) / div_factor;
    x3 = (x3 + bias) / div_factor;
    x4 = (x4 + bias) / div_factor;

    x1 = __CLIP_R(x1, 127);
    x2 = __CLIP_R(x2, 127);
    x3 = __CLIP_R(x3, 127);
    x4 = __CLIP_R(x4, 127);

    return __PACK4(x1, x2, x3, x4);
}

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
                       int8_t* p_res);

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
                           int8_t* p_res);

/**
 * @brief computes dot product of the two vectors p_a and p_b without SIMD and loop unrolling
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
                       unsigned int length);

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
                  unsigned int length);


#endif//__CL_FUNC_FUNCTIONAL_H__
