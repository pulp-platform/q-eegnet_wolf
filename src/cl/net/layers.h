/**
 * @file layers.h
 * @author Tibor Schneider
 * @date 2020/01/24
 * @brief This file contains the definitions for all the main layer functions
 */

#ifndef __CL_NET_LAYERS_H__
#define __CL_NET_LAYERS_H__

/**
 * @brief Execute the 1st layer
 * 
 * This layer does the following operation on the data:
 * 1. Convolution in time, with NET_F1 different filters of length 64, applied on all channels equally.
 * 2. Apply Batch Normalization
 *
 * @warning p_result must already be allocated on L2!
 *
 * @info The output be allocated to NET_F1 * NET_C_ALIGN * NET_T_ALIGN, because it will be flipped inplace afterwards.
 *
 * @param p_data Pointer to the input data, of shape [NET_C, NET_T], aligned to [NET_C, NET_T_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F1, NET_C, NET_T] aligned to [NET_F1, NET_C_ALIGN, NET_T_ALIGN].
 */
void net_layer1(const int8_t* p_data, int8_t* p_result);

/**
 * @brief Execute the 2nd layer
 * 
 * This layer does the following operation on the data:
 * 1. Transpose the input into the shape [NET_F1, NET_T, NET_C_ALIGN]
 * 2. Depthwise convolution in space, with NET_D filters per NET_F1, the same filter for each time sample
 * 3. Transpose the data back to [NET_F2, NET_T_ALIGN]
 * 4. Apply Batch Normalization
 * 5. Apply ReLU
 * 6. Apply Avg Pooling with kernel (1, 8)
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F1, NET_C, NET_T], aligned to [NET_F1, NET_C, NET_T_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F2, NET_T8] aligned to [NET_F2, NET_T8_ALIGN]
 */
void net_layer2(const int8_t* p_data, int8_t * p_result);

/**
 * @brief Execute the 3rd layer
 * 
 * This layer does the following operation on the data:
 * 1. Depthwise convolution in time, with 1 filter per NET_F2 of length 16.
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F2, NET_T8], aligned to [NET_F2, NET_T8_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F2, NET_T8] aligned to [NET_F2, NET_T8_ALIGN]
 */
void net_layer3(const int8_t* p_data, int8_t * p_result);

/**
 * @brief Execute the 4th layer
 * 
 * This layer does the following operation on the data:
 * 1. Pointwise Convolution, with F2 * F2 filters
 * 2. Apply Batch Normalization
 * 3. Apply ReLU
 * 4. Apply average pooling with kernel size (1, 8)
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F2, NET_T8], aligned to [NET_F2, NET_T8_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F2, NET_T64] aligned to [NET_F2, NET_T64_ALIGN]
 */
void net_layer4(const int8_t* p_data, int8_t * p_result);

/**
 * @brief Execute the 5th layer
 * 
 * This layer does the following operation on the data:
 * 1. Apply linear layer
 *
 * @param p_data Pointer to the input data, of shape [NET_F2, NET_T64], aligned to [NET_F2, NET_T64_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_D]
 */
void net_layer5(const int8_t* p_data, int8_t * p_result);

#endif//__CL_NET_LAYERS_H__
