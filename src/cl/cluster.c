#include "rt/rt_api.h"
#include "stdio.h"
#include "cluster.h"
#include "input.h"
#include "net/model.h"
#include "net/net.h"

/** 
 * \brief Cluster entry point (main)
 */
void cluster_entry(void *arg)
{
    printf("cl::cluster::cluster_entry (core %d)\n", rt_core_id());

    // allocate output memory
    int8_t * _p_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_N);

    // compute the model

#ifdef DUPLICATE_FEATUREMAP
    net_model_compute(input_data_pad, _p_output);
#else//DUPLICATE_FEATUREMAP
    net_model_compute(input_data, _p_output);
#endif//DUPLICATE_FEATUREMAP

    // print the result
    printf("Result:\n");
    for (int i = 0; i < NET_N; i++) {
        printf("Class %d: %d\n", i + 1, _p_output[i]);
    }

    // free memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)_p_output, sizeof(int8_t) * NET_N);
}
