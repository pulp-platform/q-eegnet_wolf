#include "rt/rt_api.h"
#include "stdio.h"
#include "cluster.h"
#include "input.h"
#include "net/layers.h"
#include "net/net.h"

/** 
 * \brief Cluster entry point (main)
 */
void cluster_entry(void *arg)
{
    printf("cl::cluster::cluster_entry (core %d)\n", rt_core_id());

    /*
     * Layer 1
     */

    // allocate data for result
    int8_t * p_l1_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F1 * NET_C_ALIGN * NET_T_ALIGN);

    // compute layer 1
    net_layer1(input_data, p_l1_output);

    // flip the dimension
    net_layer1_flip_inplace(p_l1_output);

    /*
     * Layer 2
     */

    // allocate memory
    int8_t * p_l2_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    // compute layer 2
    net_layer2(p_l1_output, p_l2_output);

    // free l1 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)p_l1_output, sizeof(int8_t) * NET_F1 * NET_C * NET_T_ALIGN);

    /*
     * Layer 3
     */

    // allocate memory
    int8_t * p_l3_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    // compute layer 3
    net_layer3(p_l2_output, p_l3_output);

    // flip the dimension
    net_layer3_flip_inplace(p_l3_output);

    // free l2 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)p_l2_output, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    // free l3 memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)p_l3_output, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
}
