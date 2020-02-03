/**
 * @file layer1.c
 * @author Tibor Schneider
 * @date 2020/01/29
 * @brief This file contains the Implementation for the first layer
 */

#include "rt/rt_api.h"
#include "layers.h"
#include "net.h"
#include "../func/functional.h"

#ifdef PARALLEL

#ifndef NUM_WORKERS
#define NUM_WORKERS 8
#endif

typedef struct
{
    int8_t* p_data;    // pointer to entire data vector on L1
    int8_t* p_weight;  // pointer to entire weight vector on L1
    int32_t* p_factor; // pointer to all factors on L1
    int32_t* p_offset; // pointer to all offsets on L1
    int8_t* p_thread_data; // pointer to thread local data
    int8_t* p_result;  // pointer to result on L2
} _net_layer1_kernel_t;

/**
 * @brief Layer1 kernel (convolves an output channel)
 */
void _net_layer1_kernel(void* args) {

    // get core id
    unsigned int core_id = rt_core_id();

    // extract parameters
    int8_t* _p_data = ((_net_layer1_kernel_t*)args)->p_data;
    int8_t* _p_weight = ((_net_layer1_kernel_t*)args)->p_weight;
    int32_t* _p_factor = ((_net_layer1_kernel_t*)args)->p_factor;
    int32_t* _p_offset = ((_net_layer1_kernel_t*)args)->p_offset;
    int8_t* _p_thread_data = (((_net_layer1_kernel_t*)args)->p_thread_data) + core_id * NET_T_ALIGN;
    int8_t* _p_result = ((_net_layer1_kernel_t*)args)->p_result;

    int8_t* _p_data_iter;
    int8_t* _p_weight_iter;
    int8_t* _p_result_iter;
    int32_t _factor;
    int32_t _offset;

    unsigned int _iter = core_id;
    unsigned int _k, _ch;

    rt_dma_copy_t _copy;

    // loop until all elements are computed
    while (_iter < NET_F1 * NET_C) {

        _k = _iter / NET_C;
        _ch = _iter % NET_C;

        _p_data_iter = _p_data + _ch * NET_L1_PAD_INPUT_LEN_ALIGN;
        _p_weight_iter = _p_weight + _k * NET_L1_WEIGHT_LEN;
        _p_result_iter = _p_result + (_k * NET_C_ALIGN + _ch) * NET_T_ALIGN;
        _factor = _p_factor[_k];
        _offset = _p_offset[_k];

        // convolve and scale the data (always the correct parts)
#ifdef CROSS_CORRELATE
        func_xcorr_scale(_p_data_iter, NET_L1_PAD_INPUT_LEN,
                         _p_weight_iter, NET_L1_WEIGHT_LEN,
                         _factor, _offset, _p_thread_data);
#else //CROSS_CORRELATE
        func_conv_scale(_p_data_iter, NET_L1_PAD_INPUT_LEN,
                        _p_weight_iter, NET_L1_WEIGHT_LEN,
                        _factor, _offset, _p_thread_data);
#endif //CROSS_CORRELATE

        rt_team_critical_enter();
        // copy back the results
        rt_dma_memcpy((unsigned int)_p_result_iter,
                      (unsigned int)_p_thread_data,
                      sizeof(int8_t) * NET_T,
                      RT_DMA_DIR_LOC2EXT, 0, &_copy);
        rt_dma_wait(&_copy);
        rt_team_critical_exit();

        _iter += NUM_WORKERS;

    }

    // wait for all workers to finish
    rt_team_barrier();
}
#endif

/**
 * @brief Execute the 1st layer
 * 
 * This layer does the following operation on the data:
 * 1. Convolution in time, with NET_F1 different filters of length 64, applied on all channels equally.
 * 2. Apply Batch Normalization
 *
 * @warning p_result must already be allocated on L2!
 *
 * @info The output be allocated to NET_F1 * NET_C_ALIGN * NET_T_ALIGN, because it will be flipped inplace afterwards.
 *
 * @param p_data Pointer to the input data, of shape [NET_C, NET_T], aligned to [NET_C, NET_T_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F1, NET_C, NET_T] aligned to [NET_F1, NET_C_ALIGN, NET_T_ALIGN].
 */
void net_layer1(const int8_t* p_data, int8_t* p_result) {

#ifdef PARALLEL

    const int8_t* _p_data_iter = p_data; // only used for data loading
    int8_t* _p_result_iter = p_result; // iterator over the result location

    // allocate memory for two results and two inputs
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_C * NET_L1_PAD_INPUT_LEN_ALIGN);
    int8_t* _p_weight_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F1 * NET_L1_WEIGHT_LEN);
    int8_t* _p_thread_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NUM_WORKERS * NET_T_ALIGN);
    int32_t* _p_factor_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F1);
    int32_t* _p_offset_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F1);

    rt_dma_copy_t _copy;

    // iterator over the local data
    int8_t* _p_data_loc_iter = _p_data_loc;

    // load every input vector into memory (correctly padded) and add zero padding
    for (int _ch = 0; _ch < NET_C; _ch++) {

        // add zero padding for the current vector
        int32_t* _p_pad_iter = (int32_t*)_p_data_loc_iter;
        for (int _i = 0; _i < (NET_L1_PAD_START + 3) / 4; _i++) {
            *(_p_pad_iter++) = 0;
        }
        _p_pad_iter = (int32_t*)(_p_data_loc_iter + NET_L1_PAD_INPUT_LEN_ALIGN - 4);
        // First part: aligned padding length, second part: remainder of entire padded vector
        for (int _i = 0; _i < (NET_L1_PAD_END + 3) / 4 + (NET_L1_PAD_INPUT_LEN % 4 + 3) / 4; _i++) {
            *(_p_pad_iter--) = 0;
        }

        // start the DMA transfer
        int merge = _ch == 0 ? 0 : 1;
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)(_p_data_loc_iter + NET_L1_PAD_START),
                      sizeof(int8_t) * NET_T_ALIGN,
                      RT_DMA_DIR_EXT2LOC, merge, &_copy);

        // move to the next channel
        _p_data_iter += NET_T_ALIGN;
        _p_data_loc_iter += NET_L1_PAD_INPUT_LEN_ALIGN;
    }

    // load all the weights
#ifdef CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l1_weight_reverse,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F1 * NET_L1_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
#else //CROSS_CORRELATE
    rt_dma_memcpy((unsigned int)net_l1_weight,
                  (unsigned int)_p_weight_loc,
                  sizeof(int8_t) * NET_F1 * NET_L1_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
#endif //CROSS_CORRELATE

    rt_dma_memcpy((unsigned int)net_l1_factor,
                  (unsigned int)_p_factor_loc,
                  sizeof(int32_t) * NET_F1,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_offset,
                  (unsigned int)_p_offset_loc,
                  sizeof(int32_t) * NET_F1,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    // wait until all dma transfers of the input data is complete
    rt_dma_wait(&_copy);

    // prepare the arguments for the cluster
    _net_layer1_kernel_t args;
    args.p_data = _p_data_loc;
    args.p_weight = _p_weight_loc;
    args.p_factor = _p_factor_loc;
    args.p_offset = _p_offset_loc;
    args.p_thread_data = _p_thread_data_loc;
    args.p_result = p_result;

    // call the cluster
    rt_team_fork(NUM_WORKERS, _net_layer1_kernel, (void*)(&args));

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_C * NET_L1_PAD_INPUT_LEN_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_thread_data_loc, sizeof(int8_t) * NUM_WORKERS * NET_T_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_F1 * NET_L1_WEIGHT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_loc, sizeof(int32_t) * NET_F1);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_loc, sizeof(int32_t) * NET_F1);

#else //PARALLEL

    const int8_t* _p_data_iter = p_data;
#ifdef CROSS_CORRELATE
    const int8_t* _p_weight_iter = net_l1_weight_reverse;
#else //CROSS_CORRELATE
    const int8_t* _p_weight_iter = net_l1_weight;
#endif
    int8_t* _p_result_iter = p_result;

    /*
     * Just copy the files, compute and copy the files back
     */

    // allocate memory for two results and two inputs
    int8_t * _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_L1_PAD_INPUT_LEN_ALIGN);
    int8_t * _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T_ALIGN);
#ifndef INTRINSIC_SCALE
    int32_t * _p_conv_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_T);
#endif
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
        rt_dma_wait(&_copy);

        // reset the current data pointer back to the first channel
        _p_data_iter = p_data;

        // set the result pointer to point to the start of the next k in F1:
        _p_result_iter = p_result + _k * NET_C_ALIGN * NET_T_ALIGN;

        // loop over all input channels
        for (int _ch = 0; _ch < NET_C; _ch++) {

            // copy the data
            rt_dma_memcpy((unsigned int)_p_data_iter,
                          (unsigned int)(_p_data_loc + NET_L1_PAD_START),
                          sizeof(int8_t) * NET_T,
                          RT_DMA_DIR_EXT2LOC, 0, &_copy);
            rt_dma_wait(&_copy);

#ifdef INTRINSIC_SCALE
#ifdef CROSS_CORRELATE
            // corss correlate and scale the data (always the correct parts)
            func_xcorr_scale(_p_data_loc, NET_L1_PAD_INPUT_LEN,
                             _p_weight_loc, NET_L1_WEIGHT_LEN,
                             _convert_factor, _convert_offset,
                             _p_result_loc);
#else //CROSS_CORRELATE
            // convolve and scale the data (always the correct parts)
            func_conv_scale(_p_data_loc, NET_L1_PAD_INPUT_LEN,
                            _p_weight_loc, NET_L1_WEIGHT_LEN,
                            _convert_factor, _convert_offset,
                            _p_result_loc);
#endif //CROSS_CORRELATE
#else  //INTRINSIC_SCALE
#ifdef CROSS_CORRELATE
            // convolve the data (always the correct parts)
            func_xcorr(_p_data_loc, NET_L1_PAD_INPUT_LEN,
                       _p_weight_loc, NET_L1_WEIGHT_LEN,
                       _p_conv_result_loc);
#else //CROSS_CORRELATE
            // convolve the data (always the correct parts)
            func_conv(_p_data_loc, NET_L1_PAD_INPUT_LEN,
                      _p_weight_loc, NET_L1_WEIGHT_LEN,
                      _p_conv_result_loc);
#endif //CROSS_CORRELATE

            // scale the data and pack it back to 8bit
            func_transform_32to8_bias(_p_conv_result_loc, NET_T,
                                      _convert_factor, _convert_offset, 1,
                                      _p_result_loc);
#endif

            // copy back the results
            rt_dma_memcpy((unsigned int)_p_result_iter,
                          (unsigned int)_p_result_loc,
                          sizeof(int8_t) * NET_T_ALIGN,
                          RT_DMA_DIR_LOC2EXT, 0, &_copy);
            rt_dma_wait(&_copy);

            // increment the data and results iterator
            _p_data_iter += NET_T_ALIGN;
            _p_result_iter += NET_T_ALIGN;
        }

        // increment the current weight pointer
        _p_weight_iter += NET_L1_WEIGHT_LEN;

    }

    // free up the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_L1_PAD_INPUT_LEN_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_T_ALIGN);
#ifndef INTRINSIC_SCALE
    rt_free(RT_ALLOC_CL_DATA, _p_conv_result_loc, sizeof(int32_t) * NET_T);
#endif
    rt_free(RT_ALLOC_CL_DATA, _p_weight_loc, sizeof(int8_t) * NET_L1_WEIGHT_LEN);

#endif //PARALLEL

}

/**
 * @brief Flip the C and T dimension inplace after layer 1, before layer 2
 * p_data will be of shape [NET_F1, NET_T_ALIGN, NET_C_ALIGN] afterwards.
 *
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_F1, NET_C, NET_T], aligned to [NET_F1, NET_C_ALIGN, NET_T_ALIGN]
 */
void net_layer1_flip_inplace(int8_t* p_data) {

    /*
     * For every k in F1, split the image into chunks and compute those separately
     * To do the operation inline, for each k, we copy the entire input into local memory
     */

    const int8_t* _p_data_iter = p_data; // pointer to fetch the data

    // allocate memory
    int8_t * _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_C * NET_T_ALIGN);
    int8_t * _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_T * NET_C_ALIGN);

    if (_p_data_loc == NULL || _p_result_loc == NULL) {
        printf("Error: Not enough L1 memory");
        return;
    }

    rt_dma_copy_t _copy;

    for (int _k = 0; _k < NET_F1; _k++) {
        // copy the input data to local l1 memory
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)_p_data_loc,
                      sizeof(int8_t) * NET_C * NET_T_ALIGN,
                      RT_DMA_DIR_EXT2LOC, 0, &_copy);
        rt_dma_wait(&_copy);

        // flip
#ifdef PARALLEL
        func_flip_2d_axis_par(_p_data_loc, NET_C, NET_T, _p_result_loc);
#else
        func_flip_2d_axis(_p_data_loc, NET_C, NET_T, _p_result_loc);
#endif

        // copy the results back
        rt_dma_memcpy((unsigned int)_p_data_iter,
                      (unsigned int)_p_result_loc,
                      sizeof(int8_t) * NET_T * NET_C_ALIGN,
                      RT_DMA_DIR_LOC2EXT, 0, &_copy);
        rt_dma_wait(&_copy);

        _p_data_iter += NET_C_ALIGN * NET_T_ALIGN;
    }

    // allocate memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_C * NET_T_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_T * NET_C_ALIGN);

}
