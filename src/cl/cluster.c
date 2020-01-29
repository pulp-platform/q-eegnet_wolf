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
    int8_t * p_l1_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F1 * NET_C * NET_T_ALIGN);


    // compute layer 1
    net_layer1(input_data, p_l1_output);

    // allocate data for result
    rt_free(RT_ALLOC_L2_CL_DATA, (void*) p_l1_output, sizeof(int8_t) * NET_F1 * NET_C * NET_T_ALIGN);
}
