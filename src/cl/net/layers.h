/**
 * @file layers.h
 * @author Tibor Schneider
 * @date 2020/01/24
 * @brief This file contains the definitions for all the main layer functions
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
 * @brief Flip the C and T dimension inplace after layer 1, before layer 2
 * p_data will be of shape [NET_F1, NET_T_ALIGN, NET_C_ALIGN] afterwards.
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F1, NET_C, NET_T], aligned to [NET_F1, NET_C_ALIGN, NET_T_ALIGN]
 */
void net_layer1_flip_inplace(int8_t* p_data);

/**
 * @brief Execute the 2nd layer (flipped input dimensions)
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
 * @brief Execute the 1st and the 2nd layer
 * 
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_C, NET_T], aligned to [NET_C, NET_T_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F2, NET_T8] aligned to [NET_F2, NET_T8_ALIGN].
 */
void net_fused_layer_1_2(const int8_t* p_data, int8_t* p_result);

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
 * @brief Flip the F2 and T//8 dimension inplace after layer 3, before layer 4
 * p_data will be of shape [NET_T8_ALIGN, NET_F2] afterwards.
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F2, NET_T8_ALIGN], aligned to [NET_T8_ALIGN, NET_F2]
 */
void net_layer3_flip_inplace(int8_t* p_data);

/**
 * @brief Execute the 4th layer (flipped input dimensions)
 * 
 * This layer does the following operation on the data:
 * 1. Pointwise Convolution, with F2 * F2 filters (this is a dot product when the dimensions are flipped)
 * 2. Apply Batch Normalization
 * 3. Apply ReLU
 * 4. Apply average pooling with kernel size (1, 8)
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_T8, NET_F2]
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
 * @param p_result Pointer to the output data of shape [NET_N]
 */
void net_layer5(const int8_t* p_data, int8_t * p_result);

#endif//__CL_NET_LAYERS_H__
