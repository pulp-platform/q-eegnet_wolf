/**
 * @file model.c
 * @author Tibor Schneider
 * @date 2020/02/01
 * @brief This file contains the implementation for the main model function
 */

#include "rt/rt_api.h"
#include "model.h"
#include "layers.h"
#include "net.h"

/**
 * @brief computes the output of the entire model
 *
 * @warning p_output must already be allocated on L2 memory
 *
 * @param p_data Pointer to input data on L2 memory, of shape [NET_C, NET_T], aligned to [NET_C, NET_T_ALIGN]
 *               If DUPLICATE_FEATUREMAP is enabled, the data must be padded, of shape [NET_C, NET_L1_PAD_INPUT_LEN]
 * @param p_output Pointer to output data, allocated on L2 memory, of shape [NET_N]
 */
void net_model_compute(const int8_t* p_data, int8_t* p_output) {

    /*
     * Layer 1
     */

#ifdef FUSE_LAYERS

    int8_t * _p_l2_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    net_fused_layer_1_2(p_data, _p_l2_output);

#else //FUSE_LAYERS
    // allocate data for result
    int8_t * _p_l1_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F1 * NET_C_ALIGN * NET_T_ALIGN);

    // compute layer 1
    net_layer1(p_data, _p_l1_output);

#ifdef FLIP_LAYERS
    // flip the dimension
    net_layer1_flip_inplace(_p_l1_output);
#endif //FLIP_LAYERS

    /*
     * Layer 2
     */

    // allocate memory
    int8_t * _p_l2_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    // compute layer 2
    net_layer2(_p_l1_output, _p_l2_output);

    // free l1 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)_p_l1_output, sizeof(int8_t) * NET_F1 * NET_C_ALIGN * NET_T_ALIGN);

#endif //FUSE_LAYERS

    /*
     * Layer 3
     */

    // allocate memory
    int8_t * _p_l3_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    // compute layer 3
    net_layer3(_p_l2_output, _p_l3_output);

#ifdef FLIP_LAYERS
    // flip the dimension
    net_layer3_flip_inplace(_p_l3_output);
#endif //FLIP_LAYERS

    // free l2 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)_p_l2_output, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    /*
     * Layer 4
     */

    // allocate memory
    int8_t * _p_l4_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);

    // compute layer 4
    net_layer4(_p_l3_output, _p_l4_output);

    // free l3 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)_p_l3_output, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    /*
     * Layer 5
     */

    // compute layer 5
    net_layer5(_p_l4_output, p_output);

    // free l4 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)_p_l4_output, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
}
