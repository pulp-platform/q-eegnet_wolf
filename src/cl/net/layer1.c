/**
 * @file layer1.c
 * @author Tibor Schneider
 * @date 2020/01/29
 * @brief This file contains the Implementation for the first layer
 */

//TODO remove
#include "stdio.h"

#include "rt/rt_api.h"
#include "layers.h"
#include "net.h"
#include "../func/functional.h"

/**
 * @brief Execute the 1st layer
 * 
 * This layer does the following operation on the data:
 * 1. Convolution in time, with NET_F1 different filters of length 64, applied on all channels equally.
 * 2. Apply Batch Normalization
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_C, NET_T], aligned to [NET_C, NET_T_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F1, NET_C, NET_T] aligned to [NET_F1, NET_C, NET_T_ALIGN]
 */
void net_layer1(const int8_t* p_data, int8_t* p_result) {

    printf("cl::net::net_layer1()\n");

    const int8_t* _p_data_iter = p_data;
    const int8_t* _p_weight_iter = net_l1_weight;
    int8_t* _p_result_iter = p_result;

    /*
     * Just copy the files, compute and copy the files back
     */

    // allocate memory for two results and two inputs
    int8_t * _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_L1_PAD_INPUT_LEN_ALIGN);
    int8_t * _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T_ALIGN);
    int32_t * _p_conv_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_T);
    int8_t * _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_L1_WEIGHT_LEN);

    rt_dma_copy_t _copy;

    // initialize input to have zero padding
    // initialize padding start (with loop unrolling)
    int32_t* tmp__p_data_loc = (int32_t*)(_p_data_loc);
    for (unsigned int i = 0; i < (NET_L1_PAD_START + 7) / 8; i++) {
        *(tmp__p_data_loc++) = 0;
        *(tmp__p_data_loc++) = 0;
    }
    // initialize padding end (with loop unrolling)
    tmp__p_data_loc = (int32_t*)(_p_data_loc + NET_L1_PAD_INPUT_LEN_ALIGN - 4);
    for (unsigned int i = 0; i < (NET_L1_PAD_END + 7) / 8; i++) {
        *(tmp__p_data_loc--) = 0;
        *(tmp__p_data_loc--) = 0;
    }

    // start the main loop
    for (int _k = 0; _k < NET_F1; _k++) {
        // load scale factor and offset
        int32_t _convert_factor = net_l1_factor[_k];
        int32_t _convert_offset = net_l1_offset[_k];

        // load the weights
        rt_dma_memcpy((unsigned int)_p_weight_iter,
                      (unsigned int)_p_weight_loc,
                      sizeof(int8_t) * NET_L1_WEIGHT_LEN,
                      RT_DMA_DIR_EXT2LOC, 0, &_copy);
        // increment the current weight pointer
        _p_weight_iter += NET_L1_WEIGHT_LEN;
        rt_dma_wait(&_copy);

        // reset the current data pointer back to the first channel
        _p_data_iter = p_data;

        // loop over all input channels
        for (int _ch = 0; _ch < NET_C; _ch++) {

            // copy the data
            rt_dma_memcpy((unsigned int)_p_data_iter,
                          (unsigned int)(_p_data_loc + NET_L1_PAD_START),
                          sizeof(int8_t) * NET_T,
                          RT_DMA_DIR_EXT2LOC, 0, &_copy);
            _p_data_iter += NET_T_ALIGN;
            rt_dma_wait(&_copy);

            // convolve the data (always the correct parts)
            func_conv(_p_data_loc, NET_L1_PAD_INPUT_LEN,
                      _p_weight_loc, NET_L1_WEIGHT_LEN,
                      _p_conv_result_loc);

            // scale the data and pack it back to 8bit
            func_transform_32to8_bias(_p_conv_result_loc, NET_T,
                                      _convert_factor, _convert_offset, 1,
                                      _p_result_loc);

            // copy back the results
            rt_dma_memcpy((unsigned int)_p_result_iter,
                          (unsigned int)_p_result_loc,
                          sizeof(int8_t) * NET_T_ALIGN,
                          RT_DMA_DIR_LOC2EXT, 0, &_copy);
            _p_result_iter += NET_T_ALIGN;
            rt_dma_wait(&_copy);
        }

    }

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, (void*)_p_data_loc, sizeof(int8_t) * NET_L1_PAD_INPUT_LEN_ALIGN * 2);
    rt_free(RT_ALLOC_CL_DATA, (void*)_p_result_loc, sizeof(int8_t) * NET_T_ALIGN * 2);
    rt_free(RT_ALLOC_CL_DATA, (void*)_p_conv_result_loc, sizeof(int32_t) * NET_T);
    rt_free(RT_ALLOC_CL_DATA, (void*)_p_weight_loc, sizeof(int8_t) * NET_L1_WEIGHT_LEN * 2);

}
