/**
 * @file layer4.c
 * @author Tibor Schneider
 * @date 2020/01/31
 * @brief This file contains the Implementation for the forth layer
 */

#include "rt/rt_api.h"
#include "layers.h"
#include "net.h"
#include "../func/functional.h"

#ifdef PARALLEL

#ifndef NUM_WORKERS
#define NUM_WORKERS 8
#endif

typedef struct {
    int8_t* p_data;
    int8_t* p_result;
    int8_t* p_weight;
    int32_t* p_factor;
    int32_t* p_offset;
    int32_t* p_thread_data;
} _net_layer4_kernel_t;

/**
 * @brief kernel for the parallel layer 4 implementation
 */
void _net_layer4_kernel(void* args) {

    unsigned int _core_id = rt_core_id();

    // get values from args
    _net_layer4_kernel_t* _args = args;

    int8_t* _p_data = _args->p_data;
    int8_t* _p_result = _args->p_result;
    int8_t* _p_weight = _args->p_weight;
    int32_t* _p_factor_iter = _args->p_factor;
    int32_t* _p_offset_iter = _args->p_offset;
    int32_t* _p_thread_data = _args->p_thread_data;

    // go to the correct position for the thread
    _p_weight += _core_id * NET_L4_WEIGHT_LEN;
    _p_result += _core_id * NET_T64_ALIGN;
    _p_thread_data += _core_id * NET_T64;
    _p_factor_iter += _core_id;
    _p_offset_iter += _core_id;

    int32_t* _p_thread_data_iter;
    int8_t* _p_data_iter = _p_data;
    int8_t* _p_weight_iter = _p_weight;
    int8_t* _p_result_iter = _p_result;

    int32_t _factor;
    int32_t _offset;
    int32_t _relu_threshold;
    int32_t _elem; // stores the current element, for doing dot product and ReLU
    int32_t _sum;  // stores the sum for the pooling

    unsigned int _k = _core_id;

    while (_k < NET_F2) {

        _factor = *_p_factor_iter;
        _offset = *_p_offset_iter;
        _relu_threshold = -(_offset >> 3);

        _p_data_iter = _p_data;
        _p_thread_data_iter = _p_thread_data;

        // iterate over all output time samples
        for (int _t_out = 0; _t_out < NET_T64; _t_out++) {

            // reset the sum
            _sum = 0;

            // iterate over the local environment
            for (int _t_pool = 0; _t_pool < 8; _t_pool++) {

                // compute the dot product
                _elem = func_dotp(_p_data_iter, _p_weight_iter, NET_F2);

                // do the ReLU
                _elem = __MAX(_elem, _relu_threshold);

                // add the element to the sum
                _sum += _elem;

                // go to the next input row
                _p_data_iter += NET_F2;
            }

            // store the result
            *(_p_thread_data_iter++) = _sum;

        }

        // now, we have computed the entire temporary output
        func_transform_32to8_bias(_p_thread_data, NET_T64, _factor, _offset, 1, _p_result_iter);

        // go to the next _k (for this core)
        _k += NUM_WORKERS;
        _p_weight_iter += NUM_WORKERS * NET_L4_WEIGHT_LEN;
        _p_result_iter += NUM_WORKERS * NET_T64_ALIGN;
        _p_factor_iter += NUM_WORKERS;
        _p_offset_iter += NUM_WORKERS;

    }

    // wait for all threads to finish
    rt_team_barrier();

}

#endif

/**
 * @brief Execute the 4th layer (flipped input dimensions)
 * 
 * This layer does the following operation on the data:
 * 1. Pointwise Convolution, with F2 * F2 filters (this is a dot product when the dimensions are flipped)
 * 2. Apply Batch Normalization
 * 3. Apply ReLU
 * 4. Apply average pooling with kernel size (1, 8)
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_T8, NET_F2]
 * @param p_result Pointer to the output data of shape [NET_F2, NET_T64] aligned to [NET_F2, NET_T64_ALIGN]
 */
void net_layer4(const int8_t* p_data, int8_t * p_result) {

#ifdef FLIP_LAYERS

#ifdef PARALLEL

    // we can keep everything in l1, because the data is so small. (data: 1k, result: 0.25k)
    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T8_ALIGN * NET_F2);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
    int32_t* _p_tmp_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NUM_WORKERS * NET_T64);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_F2);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    rt_dma_copy_t _copy;

    // copy all the weights at once, because copying 6 words would generate too much overhead
    rt_dma_memcpy((unsigned int)net_l4_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);

    // copy all factors
    rt_dma_memcpy((unsigned int)net_l4_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    // copy all offsets
    rt_dma_memcpy((unsigned int)net_l4_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    // copy all the data at once
    rt_dma_memcpy((unsigned int)p_data,
                  (unsigned int)_p_data_loc,
                  sizeof(int8_t) * NET_F2 * NET_T8_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_wait(&_copy);

    // prepare the kernel
    _net_layer4_kernel_t _args;
    _args.p_data = _p_data_loc;
    _args.p_result = _p_result_loc;
    _args.p_weight = _p_weight_loc;
    _args.p_factor = _p_factor_loc;
    _args.p_offset = _p_offset_loc;
    _args.p_thread_data = _p_tmp_result_loc;

    // call the kernel
    rt_team_fork(NUM_WORKERS, _net_layer4_kernel, &_args);

    // copy back the results
    rt_dma_memcpy((unsigned int)p_result,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_F2 * NET_T64_ALIGN,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_T8_ALIGN * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_tmp_result_loc, sizeof(int32_t) * NUM_WORKERS * NET_T64_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);

#else //PARALLEL

    // we can keep everything in l1, because the data is so small. (data: 1k, result: 0.25k)
    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T8_ALIGN * NET_F2);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
    int32_t* _p_tmp_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_T64);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_F2);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    rt_dma_copy_t _copy;

    // copy all the weights at once, because copying 6 words would generate too much overhead
    rt_dma_memcpy((unsigned int)net_l4_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    // copy all factors
    rt_dma_memcpy((unsigned int)net_l4_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    // copy all offsets
    rt_dma_memcpy((unsigned int)net_l4_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    // copy all the data at once
    rt_dma_memcpy((unsigned int)p_data,
                  (unsigned int)_p_data_loc,
                  sizeof(int8_t) * NET_F2 * NET_T8_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    // set up the iterators
    int8_t* _p_data_loc_iter;
    int8_t* _p_result_loc_iter = _p_result_loc;
    int32_t* _p_tmp_result_loc_iter;
    int8_t* _p_weight_loc_iter = _p_weight_loc;
    int32_t* _p_factor_loc_iter = _p_factor_loc;
    int32_t* _p_offset_loc_iter = _p_offset_loc;

    // variables needed for the computation
    int32_t _relu_threshold;
    int32_t _convert_factor;
    int32_t _convert_offset;
    int32_t _elem; // stores the current element, for doing dot product and ReLU
    int32_t _sum;  // stores the sum for the pooling

    // iterate over all output channels
    for (int _k = 0; _k < NET_F2; _k++) {

        // reset the data iterator (for each output channel, we have to go over the entire data.
        _p_data_loc_iter = _p_data_loc;

        // reset the tmp result pointer
        _p_tmp_result_loc_iter = _p_tmp_result_loc;

        // prepare the convert factor, offset and relu threshold
        _convert_factor = *(_p_factor_loc_iter++);
        _convert_offset = *(_p_offset_loc_iter++);
        _relu_threshold = -(_convert_offset >> 3);

        // iterate over all output time samples
        for (int _t_out = 0; _t_out < NET_T64; _t_out++) {

            // reset the sum
            _sum = 0;

            // iterate over the local environment
            for (int _t_pool = 0; _t_pool < 8; _t_pool++) {

                // compute the dot product
                _elem = func_dotp(_p_data_loc_iter, _p_weight_loc_iter, NET_F2);

                // do the ReLU
                _elem = __MAX(_elem, _relu_threshold);

                // add the element to the sum
                _sum += _elem;

                // go to the next input row
                _p_data_loc_iter += NET_F2;
            }

            // store the result
            *(_p_tmp_result_loc_iter++) = _sum;

        }

        // now, we have computed the entire temporary output
        func_transform_32to8_bias(_p_tmp_result_loc, NET_T64, _convert_factor, _convert_offset, 1, _p_result_loc_iter);

        // go to the next set of filters
        _p_weight_loc_iter += NET_F2;
        // go to the next result part
        _p_result_loc_iter += NET_T64_ALIGN;

    }

    // copy back the results
    rt_dma_memcpy((unsigned int)p_result,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_F2 * NET_T64_ALIGN,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_T8_ALIGN * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_tmp_result_loc, sizeof(int32_t) * NET_T64_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);

#endif //PARALLEL

#else //FLIP_LAYERS

    // we can keep everything in l1, because the data is so small. (data: 1k, result: 0.25k)
    // allocate local memory
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
    int32_t* _p_tmp_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_T64);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_F2);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    rt_dma_copy_t _copy;

    // copy all the weights at once, because copying 6 words would generate too much overhead
    rt_dma_memcpy((unsigned int)net_l4_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F2 * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    // copy all factors
    rt_dma_memcpy((unsigned int)net_l4_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    // copy all offsets
    rt_dma_memcpy((unsigned int)net_l4_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    // copy all the data at once
    rt_dma_memcpy((unsigned int)p_data,
                  (unsigned int)_p_data_loc,
                  sizeof(int8_t) * NET_F2 * NET_T8_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_wait(&_copy);

    // set up the iterators
    int8_t* _p_data_loc_iter;
    int8_t* _p_result_loc_iter = _p_result_loc;
    int32_t* _p_tmp_result_loc_iter;
    int8_t* _p_weight_loc_iter = _p_weight_loc;
    int32_t* _p_factor_loc_iter = _p_factor_loc;
    int32_t* _p_offset_loc_iter = _p_offset_loc;

    // variables needed for the computation
    int32_t _relu_threshold;
    int32_t _convert_factor;
    int32_t _convert_offset;
    int32_t _elem; // stores the current element, for doing dot product and ReLU
    int32_t _sum;  // stores the sum for the pooling

    // iterate over all output channels
    for (int _k = 0; _k < NET_F2; _k++) {

        // reset the data iterator (for each output channel, we have to go over the entire data.
        _p_data_loc_iter = _p_data_loc;

        // reset the tmp result pointer
        _p_tmp_result_loc_iter = _p_tmp_result_loc;

        // prepare the convert factor, offset and relu threshold
        _convert_factor = *(_p_factor_loc_iter++);
        _convert_offset = *(_p_offset_loc_iter++);
        _relu_threshold = -(_convert_offset >> 3);

        // iterate over all output time samples
        for (int _t_out = 0; _t_out < NET_T64; _t_out++) {

            // reset the sum
            _sum = 0;

            // iterate over the local environment
            for (int _t_pool = 0; _t_pool < 8; _t_pool++) {

                // compute the dot product
                _elem = func_dotp_slow(_p_data_loc_iter, NET_T8_ALIGN, _p_weight_loc_iter, 1, NET_F2);

                // do the ReLU
                _elem = __MAX(_elem, _relu_threshold);

                // add the element to the sum
                _sum += _elem;

                // go to the next column
                _p_data_loc_iter += 1;
            }

            // store the result
            *(_p_tmp_result_loc_iter++) = _sum;

        }

        // now, we have computed the entire temporary output
        func_transform_32to8_bias(_p_tmp_result_loc, NET_T64, _convert_factor, _convert_offset, 1, _p_result_loc_iter);

        // go to the next set of filters
        _p_weight_loc_iter += NET_F2;
        // go to the next result part
        _p_result_loc_iter += NET_T64_ALIGN;

    }

    // copy back the results
    rt_dma_memcpy((unsigned int)p_result,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_F2 * NET_T64_ALIGN,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_T8_ALIGN * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F2 * NET_T64_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_tmp_result_loc, sizeof(int32_t) * NET_T64_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F2 * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F2);

#endif

}
