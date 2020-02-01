/**
 * @file model.h
 * @author Tibor Schneider
 * @date 2020/02/01
 * @brief This file contains the definitions for the model
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
