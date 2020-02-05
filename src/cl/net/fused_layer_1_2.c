/**
 * @file fused_layer_1_2.c
 * @author Tibor Schneider
 * @date 2020/02/04
 * @brief This file contains the Implementation for the fused layer 1 and 2
 */

#include "rt/rt_api.h"
#include "layers.h"
#include "net.h"
#include "../func/functional.h"

#ifdef FUSE_LAYERS

// do checks
#ifndef PARALLEL
#error "Parallel is required to fuse layers"
#endif
#ifndef CROSS_CORRELATE
#error "Cross Correlate is required to fuse layers"
#endif
#ifndef INTRINSIC_SCALE
#error "intrinsic scale is required to fuse layers"
#endif

#ifndef NUM_WORKERS
#define NUM_WORKERS 8
#endif

// all optimizations used below require nice shapes

#if NET_F1 != NUM_WORKERS
#error "The number of spectral filters must be equal to the number of workers"
#endif

#if NET_L1_PAD_INPUT_LEN % 4 != 0
#error "The padded input length must be divisible by 4"
#endif

#if NET_T8_ALIGN != NET_T8
#error "T / 8 must be divisible by 4"
#endif

#if NET_D != 2
#error "D must be equal to 2"
#endif

#define _SHUFFLEMASK1 (v4s){1,2,3,4}
#define _SHUFFLEMASK2 (v4s){2,3,4,5}
#define _SHUFFLEMASK3 (v4s){3,4,5,6}

typedef struct {
    int8_t* p_data;
    int8_t* p_result;

    int8_t* p_weight_l1;
    int32_t* p_factor_l1;
    int32_t* p_offset_l1;

    int8_t* p_weight_l2;
    int32_t* p_factor_l2;
    int32_t* p_offset_l2;

    int8_t* p_thread_data;
} _net_fused_layer_1_2_kernel_t;


/**
 * @brief Kernel for doing the computation
 */
void _net_fused_layer_1_2_kernel(void* args) {

    unsigned int _core_id = rt_core_id();

    // get values from args
    _net_fused_layer_1_2_kernel_t* _args = args;

    int8_t* _p_data = _args->p_data;
    int8_t* _p_result = _args->p_result;
    int8_t* _p_weight_l1 = _args->p_weight_l1;
    int32_t* _p_factor_l1 = _args->p_factor_l1;
    int32_t* _p_offset_l1 = _args->p_offset_l1;
    int8_t* _p_weight_l2 = _args->p_weight_l2;
    int32_t* _p_factor_l2 = _args->p_factor_l2;
    int32_t* _p_offset_l2 = _args->p_offset_l2;
    int8_t* _p_thread_data = _args->p_thread_data;

    // change the pointers to point to the data used by the specific core
    _p_result += _core_id * 2 * NET_T8_ALIGN;
    _p_weight_l1 += _core_id * NET_L1_WEIGHT_LEN;
    _p_factor_l1 += _core_id;
    _p_offset_l1 += _core_id;
    _p_weight_l2 += _core_id * 2 * NET_L2_WEIGHT_LEN;
    _p_factor_l2 += _core_id * 2;
    _p_offset_l2 += _core_id * 2;
    _p_thread_data += _core_id * NET_C_ALIGN * 4;

    // load the scaling factors
    int32_t _factor_l1 = *_p_factor_l1;
    int32_t _offset_l1 = *_p_offset_l1;
    int32_t _factor_l2_0 = *(_p_factor_l2 + 0);
    int32_t _offset_l2_0 = *(_p_offset_l2 + 0);
    int32_t _factor_l2_1 = *(_p_factor_l2 + 1);
    int32_t _offset_l2_1 = *(_p_offset_l2 + 1);
    int32_t _threshold_0 = -(_offset_l2_0 >> 3);
    int32_t _threshold_1 = -(_offset_l2_1 >> 3);

    int8_t* _p_data_iter = _p_data; // iterator over the current elements for which we do the computation
    int8_t* _p_data_iter_comp;      // Pointer to the data while doing the dot product
    int8_t* _p_weight_l1_iter_comp; // pointer to the weights while doing the dot product
    int8_t* _p_thread_data_iter;    // iterator over the thread data
    int8_t* _p_result_iter = _p_result;

    v4s _x0, _x1, _x2, _x3;
    v4s _y;
    int32_t _acc0, _acc1, _acc2, _acc3;

    int32_t _pool_sum_0;
    int32_t _pool_sum_1;
    int32_t _elem;

    // iterate over all output samples
    for (int _t_out = 0; _t_out < NET_T8; _t_out++) {

        // reset the pooling summation register
        _pool_sum_0 = 0;
        _pool_sum_1 = 0;

        // iterate over all the padding samples divided by 4, because we compute 4 values at the same time
        for (int _t_pad = 0; _t_pad < 8 / 4; _t_pad++) {

            /*
             * compute the intermediate vector
             */

            // setup the iteration
            _p_thread_data_iter = _p_thread_data;

            for (int _ch = 0; _ch < NET_C; _ch++) {

                // setup the iteration
                _p_data_iter_comp = _p_data_iter + _ch * NET_L1_PAD_INPUT_LEN_ALIGN;
                _p_weight_l1_iter_comp = _p_weight_l1;

                _acc0 = 0;
                _acc1 = 0;
                _acc2 = 0;
                _acc3 = 0;

                // do the dot product of 4 values at the same time
                for (int _i = 0; _i < NET_L1_WEIGHT_LEN / 4; _i++) {
                    // load the data
                    _x0 = *((v4s*)(_p_data_iter_comp + 0));
                    _x3 = *((v4s*)(_p_data_iter_comp + 4));
                    _y = *((v4s*)_p_weight_l1_iter_comp);

                    _x1 = __builtin_shuffle(_x0, _x3, _SHUFFLEMASK1);
                    _x2 = __builtin_shuffle(_x0, _x3, _SHUFFLEMASK2);
                    _x3 = __builtin_shuffle(_x0, _x3, _SHUFFLEMASK3);

                    _acc0 = __SUMDOTP4(_x0, _y, _acc0);
                    _acc1 = __SUMDOTP4(_x1, _y, _acc1);
                    _acc2 = __SUMDOTP4(_x2, _y, _acc2);
                    _acc3 = __SUMDOTP4(_x3, _y, _acc3);

                    // go to the next iteration
                    _p_data_iter_comp += 4;
                    _p_weight_l1_iter_comp += 4;
                }

                // scale the values
                _acc0 = (_acc0 + _offset_l1) / _factor_l1;
                _acc1 = (_acc1 + _offset_l1) / _factor_l1;
                _acc2 = (_acc2 + _offset_l1) / _factor_l1;
                _acc3 = (_acc3 + _offset_l1) / _factor_l1;

                // clip the values
                _acc0 = __CLIP_R(_acc0, 127);
                _acc1 = __CLIP_R(_acc1, 127);
                _acc2 = __CLIP_R(_acc2, 127);
                _acc3 = __CLIP_R(_acc3, 127);

                // store the values as 1 byte in the appropriate position
                *(_p_thread_data_iter + 0 * NET_C_ALIGN) = (int8_t)_acc0;
                *(_p_thread_data_iter + 1 * NET_C_ALIGN) = (int8_t)_acc1;
                *(_p_thread_data_iter + 2 * NET_C_ALIGN) = (int8_t)_acc2;
                *(_p_thread_data_iter + 3 * NET_C_ALIGN) = (int8_t)_acc3;

                // go to the next value in the thread data
                _p_thread_data_iter++;

            }

            /*
             * Now, the temporary vector of 4 elements is computed. We now just need to do the dot product
             * The Dot product needs to be done for both filters
             * The result is summed up for padding
             */

            // TODO fuse those dot products s.t. two outputs are computed at the same time
            // first output channel
            for (int _i = 0; _i < 4; _i++) {
                // first element
                _elem = func_dotp(_p_thread_data + _i * NET_C_ALIGN, _p_weight_l2, NET_L2_WEIGHT_LEN);
                _elem = __MAX(_elem, _threshold_0);
                _pool_sum_0 += _elem;

                // second element
                _elem = func_dotp(_p_thread_data + _i * NET_C_ALIGN, _p_weight_l2 + NET_L2_WEIGHT_LEN, NET_L2_WEIGHT_LEN);
                _elem = __MAX(_elem, _threshold_1);
                _pool_sum_1 += _elem;
            }

            // move to the next 4 time samples
            _p_data_iter += 4;

        }

        // now, we have computed the temporary _pool_sum.
        // scale it
        _pool_sum_0 = (_pool_sum_0 + _offset_l2_0) / _factor_l2_0;
        _pool_sum_1 = (_pool_sum_1 + _offset_l2_1) / _factor_l2_1;

        _pool_sum_0 = __CLIP_R(_pool_sum_0, 127);
        _pool_sum_1 = __CLIP_R(_pool_sum_1, 127);

        // store the values
        *(_p_result_iter + 0 * NET_T8_ALIGN) = _pool_sum_0;
        *(_p_result_iter + 1 * NET_T8_ALIGN) = _pool_sum_1;

        // change the result iterator
        _p_result_iter++;

    }

    rt_team_barrier();

}


/**
 * @brief Execute the 1st and the 2nd layer
 * 
 * @warning p_result must already be allocated on L2!
 *
 * @param p_data Pointer to the input data, of shape [NET_C, NET_T], aligned to [NET_C, NET_T_ALIGN]
 * @param p_result Pointer to the output data of shape [NET_F2, NET_T8] aligned to [NET_F2, NET_T8_ALIGN].
 */
void net_fused_layer_1_2(const int8_t* p_data, int8_t* p_result) {

    // allocate memory for two results and two inputs
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_C * NET_L1_PAD_INPUT_LEN_ALIGN);
    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    int8_t* _p_weight_l1_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F1 * NET_L1_WEIGHT_LEN);
    int32_t* _p_factor_l1_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F1);
    int32_t* _p_offset_l1_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F1);

    int8_t* _p_weight_l2_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    int32_t* _p_factor_l2_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_l2_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    int8_t* _p_thread_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NUM_WORKERS * NET_C_ALIGN * 4);

    rt_dma_copy_t _copy;

    // iterator over the local data
    int8_t* _p_data_loc_iter = _p_data_loc;
    const int8_t* _p_data_iter = p_data; // only used for data loading

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

    // load all the weights of layer 1
    rt_dma_memcpy((unsigned int)net_l1_weight_reverse,
                  (unsigned int)_p_weight_l1_loc,
                  sizeof(int8_t) * NET_F1 * NET_L1_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_factor,
                  (unsigned int)_p_factor_l1_loc,
                  sizeof(int32_t) * NET_F1,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_offset,
                  (unsigned int)_p_offset_l1_loc,
                  sizeof(int32_t) * NET_F1,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    // load all the weights of layer 2
    rt_dma_memcpy((unsigned int)net_l2_weight,
                  (unsigned int)_p_weight_l2_loc,
                  sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l2_factor,
                  (unsigned int)_p_factor_l2_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l2_offset,
                  (unsigned int)_p_offset_l2_loc,
                  sizeof(int32_t) * NET_F2,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    // wait until all dma transfers of the input data is complete
    rt_dma_wait(&_copy);

    // now, all the data necessary for computation resides in local memory! Prepare the kernel
    _net_fused_layer_1_2_kernel_t _args;
    _args.p_data = _p_data_loc;
    _args.p_result = _p_result_loc;
    _args.p_weight_l1 = _p_weight_l1_loc;
    _args.p_factor_l1 = _p_factor_l1_loc;
    _args.p_offset_l1 = _p_offset_l1_loc;
    _args.p_weight_l2 = _p_weight_l2_loc;
    _args.p_factor_l2 = _p_factor_l2_loc;
    _args.p_offset_l2 = _p_offset_l2_loc;
    _args.p_thread_data = _p_thread_data_loc;

    // start the kernel
    rt_team_fork(NUM_WORKERS, _net_fused_layer_1_2_kernel, &_args);

    // copy all results back to the results vector
    rt_dma_memcpy((unsigned int)p_result,
                  (unsigned int)_p_result_loc,
                  sizeof(int8_t) * NET_F2 * NET_T8_ALIGN,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free all the memory
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * NET_C * NET_L1_PAD_INPUT_LEN_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    rt_free(RT_ALLOC_CL_DATA, _p_weight_l1_loc, sizeof(int8_t) * NET_F1 * NET_L1_WEIGHT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_l1_loc, sizeof(int32_t) * NET_F1);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_l1_loc, sizeof(int32_t) * NET_F1);

    rt_free(RT_ALLOC_CL_DATA, _p_weight_l2_loc, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_l2_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_l2_loc, sizeof(int32_t) * NET_F2);

    rt_free(RT_ALLOC_CL_DATA, _p_thread_data_loc, sizeof(int8_t) * NUM_WORKERS * NET_C_ALIGN * 4);

}

#endif //FUSE_LAYERS
