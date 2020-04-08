/**
 * @file model.h
 * @author Tibor Schneider
 * @date 2020/02/01
 * @brief This file contains the definitions for the model
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


#ifndef __CL_NET_MODEL_H__
#define __CL_NET_MODEL_H__

/**
 * @brief computes the output of the entire model
 *
 * @warning p_output must already be allocated on L2 memory
 *
 * @param p_data Pointer to input data on L2 memory, of shape [NET_C, NET_T], aligned to [NET_C, NET_T_ALIGN]
 * @param p_output Pointer to output data, allocated on L2 memory, of shape [NET_N]
 */
void net_model_compute(const int8_t* p_data, int8_t* p_output);

#endif//__CL_NET_MODEL_H__
