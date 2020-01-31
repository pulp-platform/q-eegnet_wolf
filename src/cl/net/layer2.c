/**
 * @file layer2.c
 * @author Tibor Schneider
 * @date 2020/01/30
 * @brief This file contains the Implementation for the first layer
 */

#include "rt/rt_api.h"
#include "layers.h"
#include "net.h"
#include "../func/functional.h"

/**
 * @brief Execute the 2nd layer
 * 
 * This layer does the following operation on the data:
 * 2. Depthwise convolution in space, with NET_D filters per NET_F1, the same filter for each time sample
 * 4. Apply Batch Normalization
 * 5. Apply ReLU
 * 6. Apply Avg Pooling with kernel (1, 8)
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F1, NET_T, NET_C], aligned to [NET_F1, NET_T, NET_C_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F2, NET_T8] aligned to [NET_F2, NET_T8_ALIGN]
 */
void net_layer2(const int8_t* p_data, int8_t * p_result) {

    /*
     * We compute one output channel (one of F2) at a time.
     */

    const int8_t* _p_data_iter = p_data;          // iterator over the current image of the input
    const int8_t* _p_result_iter = p_result;      // iterator over the current vector of the output

    rt_dma_copy_t _copy;

    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T8_ALIGN);
    int32_t* _p_tmp_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_T8_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    // copy all the weights at once, because copying 6 words would generate too much overhead
    rt_dma_memcpy((unsigned int)net_l2_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    // copy all factors
    rt_dma_memcpy((unsigned int)net_l2_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    // copy all offsets
    rt_dma_memcpy((unsigned int)net_l2_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_wait(&_copy);

    int8_t* _p_weight_loc_iter = _p_weight_loc;  // iterator over the current weights (filter)
    int32_t* _p_factor_loc_iter = _p_factor_loc; // iterator over the current factor
    int32_t* _p_offset_loc_iter = _p_offset_loc; // iterator over the current offset
    int8_t* _p_data_loc_iter;                    // iterator over the current local data row
    int32_t* _p_tmp_result_loc_iter;             // iterator over the current temporary result

    int32_t _relu_threshold;
    int32_t _convert_factor;
    int32_t _convert_offset;
    int32_t _elem; // stores the current element, for doing dot product and ReLU
    int32_t _sum;  // stores the sum for the pooling

    // loop over all input images
    for (unsigned int _k = 0; _k < NET_F1; _k++) {

        // copy the corresponding input data to local memory
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)_p_data_loc,
                      sizeof(int8_t) * NET_T * NET_C_ALIGN,
                      RT_DMA_DIR_EXT2LOC, 0, &_copy);
        rt_dma_wait(&_copy);

        // reset the local data iterator
        _p_data_loc_iter = _p_data_loc;
        
        // loop over all output filters for the corresponding input image
        for (unsigned int _i = 0; _i < 2; _i++) {

            // reset the temporary local result iterator
            _p_tmp_result_loc_iter = _p_tmp_result_loc;
            // reset the local input iterator to point to the first line
            _p_data_loc_iter = _p_data_loc;

            // compute relu threshold
            _convert_factor = *_p_factor_loc_iter++;
            _convert_offset = *_p_offset_loc_iter++;
            _relu_threshold = -(_convert_offset >> 3);

            // loop over all output time samples (after pooling)
            for (unsigned int _t_out = 0; _t_out < NET_T8; _t_out++) {

                // reset the sum
                _sum = 0;

                // loop over all 8 elements in the local neighborhood
                for (unsigned int _t_pool = 0; _t_pool < 8; _t_pool++) {

                    // do the dot product
                    // we copute the dot product over C_ALIGN instead of C, it is faster and the additional elements are 0
                    _elem = func_dotp(_p_data_loc_iter, _p_weight_loc_iter, NET_C_ALIGN);

                    // do the ReLU
                    _elem = __MAX(_elem, _relu_threshold);

                    // add the element to the sum
                    _sum += _elem;

                    // go to the next input row
                    _p_data_loc_iter += NET_C_ALIGN;
                }

                // write the result of the local pooing to temporary memory
                *(_p_tmp_result_loc_iter++) = _sum;

            }

            // now, we have computed the temporary 32bit result vector. Scale it!
            func_transform_32to8_bias(_p_tmp_result_loc, NET_T8,
                                      _convert_factor, _convert_offset, 1,
                                      _p_result_loc);

            // copy the values back to L2 memory
            rt_dma_memcpy((unsigned int)_p_result_iter,
                          (unsigned int)_p_result_loc,
                          sizeof(int8_t) * NET_T8,
                          RT_DMA_DIR_LOC2EXT, 0, &_copy);
            rt_dma_wait(&_copy);

            // go to the next output channel
            _p_result_iter += NET_T8_ALIGN;
            // use the next filter
            _p_weight_loc_iter += NET_L2_WEIGHT_LEN;

        }

        // go to the next image (next one of F1)
        _p_data_iter += NET_T_ALIGN * NET_C_ALIGN;
    }

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_T * NET_C_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_tmp_result_loc, sizeof(int32_t) * NET_T8_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);

}
