/**
 * @file functional.h
 * @author Tibor Schneider
 * @date 2020/01/25
 * @brief This file contains the definitions for all main mathematical functions
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


#endif//__CL_FUNC_FUNCTIONAL_H__