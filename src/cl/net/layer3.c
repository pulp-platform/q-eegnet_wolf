/**
 * @file layer3.c
 * @author Tibor Schneider
 * @date 2020/01/31
 * @brief This file contains the Implementation for the third layer
 */

#include "rt/rt_api.h"
#include "layers.h"
#include "net.h"
#include "../func/functional.h"

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
void net_layer3(const int8_t* p_data, int8_t * p_result) {

    /*
     * Depthwise Convoluton, compute every channel separately
     */

    const int8_t* _p_data_iter = p_data;          // iterator over the current input vector
    const int8_t* _p_result_iter = p_result;      // iterator over the current output vector

    rt_dma_copy_t _copy;

    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_L3_PAD_INPUT_LEN_ALIGN);
    int32_t* _p_tmp_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_T8);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T8_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L3_WEIGHT_LEN);

    // initialize input to have zero padding
    *((int32_t*)(_p_data_loc + 0)) = 0;
    *((int32_t*)(_p_data_loc + 4)) = 0;
    *((int32_t*)(_p_data_loc + NET_L3_PAD_INPUT_LEN_ALIGN - 4)) = 0;
    *((int32_t*)(_p_data_loc + NET_L3_PAD_INPUT_LEN_ALIGN - 8)) = 0;
    *((int32_t*)(_p_data_loc + NET_L3_PAD_INPUT_LEN_ALIGN - 12)) = 0;

    // copy all the weights at once, because we get less overhead
    rt_dma_memcpy((unsigned int)net_l3_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_L3_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    int8_t* _p_weight_loc_iter = _p_weight_loc;  // iterator over the current weights (filter)

    // loop over all channels
    for (unsigned int _k = 0; _k < NET_F2; _k++) {

        // copy the corresponding input data to local memory, keeping the padding
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)_p_data_loc + NET_L3_PAD_START,
                      sizeof(int8_t) * NET_T8,
                      RT_DMA_DIR_EXT2LOC, 0, &_copy);
        rt_dma_wait(&_copy);

        // do the convolution
        func_conv(_p_data_loc, NET_L3_PAD_INPUT_LEN, _p_weight_loc_iter, NET_L3_WEIGHT_LEN, _p_tmp_result_loc);

        // scale the values
        func_transform_32to8(_p_tmp_result_loc, NET_T8, NET_L3_FACTOR, 1, _p_result_loc);

        // copy the results back
        rt_dma_memcpy((unsigned int)_p_result_iter,
                      (unsigned int)_p_result_loc,
                      sizeof(int8_t) * NET_T8,
                      RT_DMA_DIR_LOC2EXT, 0, &_copy);
        rt_dma_wait(&_copy);

        // move the iterators to the next elements
        _p_data_iter += NET_T8_ALIGN;
        _p_result_iter += NET_T8_ALIGN;
        _p_weight_loc_iter += NET_L3_WEIGHT_LEN;

    }
}
