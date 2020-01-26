/**
 * @file layer1.c
 * @author Tibor Schneider
 * @date 2020/01/24
 * @brief This file contains the Implementation for the first layer
 */

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
 * @param p_data Pointer to the input data, of shape [NET_C, NET_T], aligned to [NET_C, NET_T_ALIGN]
 * @return Pointer to the output data of shape [NET_F1, NET_C, NET_T] aligned to [NET_F1, NET_C, NET_T_ALIGN]
 */
int8_t* net_layer1(int8_t* p_data) {

    // Allocate the output memory on L2 Memory
    int8_t * p_result = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int8_t) * NET_F1 * NET_C * NET_T_ALIGN);

#ifdef PARALLEL
#error "Not Implemented"
#else//PARALLEL
    
    /*
     * Compute every Channel for every Filter separately
     */

    // allocate memory for two results and two inputs
    int8_t * p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_L1_PAD_INPUT_LEN_ALIGN * 2);
    int8_t * p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T_ALIGN * 2);
    int32_t * p_conv_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_T);
    int8_t * p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_L1_WEIGHT_LEN * 2);

    // initialize input to have zero padding
    for (unsigned int i = 0; i < 2; i++) {
        // initialize padding start (with loop unrolling)
        int32_t* tmp_p_data_loc = (int32_t*)(p_data_loc + i * NET_L1_PAD_INPUT_LEN_ALIGN);
        for (unsigned int j = 0; j < (NET_L1_PAD_START + 7) / 8; j++) {
            *(tmp_p_data_loc++) = 0;
            *(tmp_p_data_loc++) = 0;
        }
        // initialize padding end (with loop unrolling)
        tmp_p_data_loc = (int32_t*)(p_data_loc + (i + 1) * NET_L1_PAD_INPUT_LEN_ALIGN - 4);
        for (unsigned int j = 0; j < (NET_L1_PAD_END + 7) / 8; j++) {
            *(tmp_p_data_loc--) = 0;
            *(tmp_p_data_loc--) = 0;
        }
    }

    // copy the first data to l1 and wait until it is finished (copy and respect padding)
    rt_dma_copy_t data_copy;
    int part = 0; // 1 if we use the upper part, 0 if we use the lower part
    rt_dma_memcpy((unsigned int)p_data,
                  (unsigned int)(p_data_loc + NET_L1_PAD_START + NET_L1_PAD_INPUT_LEN_ALIGN * part),
                  sizeof(int8_t) * NET_T,
                  RT_DMA_DIR_EXT2LOC, 0, &data_copy);

    // we also need to copy the weight vector (only every time k changes
    rt_dma_copy_t weight_copy;
    int weight_part = 0; // 1 if we use the upper part, 0 if we use the lower part
    rt_dma_memcpy((unsigned int)net_l1_weight,
                  (unsigned int)(p_weight_loc + NET_L1_WEIGHT_LEN * weight_part),
                  sizeof(int8_t) * NET_L1_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 0, &weight_copy);

    // we need two variables to know wether the data is available in the correct slot
    rt_dma_copy_t result_copy[2];

    // outer loop over all different filters
    for (int k = 0; k < NET_F1; k++) {
        // load scale factor and offset
        int32_t convert_factor = net_l1_factor[k];
        int32_t convert_offset = net_l1_offset[k];
        
        // set the next weight part to the inverse of the current part
        int next_weight_part = weight_part ? 0 : 1;

        // wait until the weights are copied
        rt_dma_wait(&weight_copy);

        // copy the next set of weights
        if (k + 1 < NET_F1) {
            rt_dma_memcpy((unsigned int)(net_l1_weight + (k + 1) * NET_L1_WEIGHT_LEN),
                          (unsigned int)(p_weight_loc + NET_L1_WEIGHT_LEN * next_weight_part),
                          sizeof(int8_t) * NET_L1_WEIGHT_LEN,
                          RT_DMA_DIR_EXT2LOC, 0, &weight_copy);
        }

        //inner loop over all different channels
        for (int ch = 0; ch < NET_C; ch++) {
            // set the next_part to be the inverse of the current part
            int next_part = part ? 0 : 1;

            // wait until the data is copied
            rt_dma_wait(&data_copy);

            // copy the data for the next iteration
            int next_ch = ch + 1;
            int next_k = (next_ch >= NET_C) ? k + 1 : k;
            next_ch = (next_ch >= NET_C) ? next_ch - NET_C : next_ch;
            // only copy if there is data to process for the next iteration
            if (next_k < NET_F1) {
                rt_dma_memcpy((unsigned int)(p_data + (next_k * NET_C + next_ch) * NET_T_ALIGN),
                              (unsigned int)(p_data_loc + NET_L1_PAD_START + NET_L1_PAD_INPUT_LEN_ALIGN * next_part),
                              sizeof(int8_t) * NET_T,
                              RT_DMA_DIR_EXT2LOC, 0, &data_copy);
            }

            // wait for the output slot to be available. This is only necessary after the third iteration
            if (k >= 1 || ch >= 2) {
                rt_dma_wait(result_copy + part);
            }

            // convolve the data (always the correct parts)
            func_conv(p_data_loc + NET_L1_PAD_INPUT_LEN_ALIGN * part,
                      NET_L1_PAD_INPUT_LEN,
                      p_weight_loc + NET_L1_WEIGHT_LEN * weight_part,
                      NET_L1_WEIGHT_LEN,
                      p_conv_result_loc);

            // scale the data and pack it back to 8bit
            func_transform_32to8_bias(p_conv_result_loc, NET_T,
                                      convert_factor, convert_offset, 1,
                                      p_result_loc + NET_T_ALIGN * part);

            // start to copy back the results
            rt_dma_memcpy((unsigned int)(p_result + (k * NET_C + ch) * NET_T_ALIGN),
                          (unsigned int)p_result_loc + NET_T_ALIGN * part,
                          sizeof(int8_t) * NET_T_ALIGN,
                          RT_DMA_DIR_LOC2EXT, 0, result_copy + part);

            // set the part equal to the next part for the next iteration
            part = next_part;
        }

        // set the weight part equal to the next weight part for the next iteration
        weight_part = next_weight_part;
    }
    
#endif//PARALLEL

    return p_result;
    
}
