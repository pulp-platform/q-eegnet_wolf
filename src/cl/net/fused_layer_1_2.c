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

#ifdef NO_INTERMEDIATE_SCALE

#ifdef DUPLICATE_FEATUREMAP

// dimension the split, it is important that all parts are divisible by 8
// We split it into 5 parts, of size

#ifdef DEFAULT_DIM
#define _T_SPLIT_LEN 240
#else//DEFAULT_DIM
#define _T_SPLIT_LEN 248
#endif//DEFAULT_DIM

#define _T_SPLIT_MEM_OFFSET (4 * 0)
#define _T_SPLIT_LEN_LAST (NET_L1_PAD_INPUT_LEN - (_T_SPLIT_LEN * 4))
#define _T_SPLIT_MEM_SIZE ((_T_SPLIT_LEN > _T_SPLIT_LEN_LAST ? _T_SPLIT_LEN : _T_SPLIT_LEN_LAST) * NET_C + _T_SPLIT_MEM_OFFSET)
#if (_T_SPLIT_LEN % 8 != 0)
#error "The splits must all be of size 8!"
#endif

#define _THREAD_MEM_OFFSET 0

/*
 * Method of duplicating the featuremap 4 times and storing it on L1, shifted by 1 element
 */


/**
 * @brief this function computes the convolution of 4 values in time of all channels
 *
 * @param core_id
 * @param p_data pointer to the current data where we start the convolution, must be in the first channel
 * @param stride number of elements in a single row of data
 * @param p_weight Pointer to the weight vector
 * @param offset Amount to offset the result at the end of the computation
 * @param p_result pointer to the result data of size [4, NET_C_ALIGN], must be thread local data
 */
void _net_fused_layer_1_2_kernel_conv(unsigned int core_id,
                                      const int8_t* p_data,
                                      unsigned int stride,
                                      const int8_t* p_weight,
                                      uint32_t offset,
                                      int32_t* p_result) {

    // setup iterators
    const int8_t* _p_data_iter0;
    const int8_t* _p_data_iter1;
    const int8_t* _p_data_iter2;
    const int8_t* _p_data_iter3;
    const int8_t* _p_weight_iter = p_weight;

    // declare local variables
    int32_t _acc0, _acc1, _acc2, _acc3;

    for (int _ch_t = 0; _ch_t < NET_C; _ch_t++) {

        //int _ch = _ch_t + core_id;
        //if (_ch >= NET_C) _ch -= NET_C;
        int _ch = _ch_t;

        // setup the iteration
        _p_data_iter0 = p_data + _ch * stride;
        _p_data_iter1 = p_data + _ch * stride + 1 * _T_SPLIT_MEM_SIZE;
        _p_data_iter2 = p_data + _ch * stride + 2 * _T_SPLIT_MEM_SIZE;
        _p_data_iter3 = p_data + _ch * stride + 3 * _T_SPLIT_MEM_SIZE;
        _p_weight_iter = p_weight;

        _acc0 = offset;
        _acc1 = offset;
        _acc2 = offset;
        _acc3 = offset;

        asm volatile("lp.setup x0,%[num_t],36;"
                     "   p.lw s9,4(%[p_weight]!);"
                     "   p.lw s5,4(%[p_data0]!);"
                     "   p.lw s6,4(%[p_data1]!);"
                     "   p.lw s7,4(%[p_data2]!);"
                     "   p.lw s8,4(%[p_data3]!);"
                     "   pv.sdotsp.b %[_acc0],s5,s9;"
                     "   pv.sdotsp.b %[_acc1],s6,s9;"
                     "   pv.sdotsp.b %[_acc2],s7,s9;"
                     "36:pv.sdotsp.b %[_acc3],s8,s9;"
                     : [p_weight] "+r" (_p_weight_iter),
                       [_acc0] "+r" (_acc0),
                       [_acc1] "+r" (_acc1),
                       [_acc2] "+r" (_acc2),
                       [_acc3] "+r" (_acc3),
                       [p_data0] "+r" (_p_data_iter0),
                       [p_data1] "+r" (_p_data_iter1),
                       [p_data2] "+r" (_p_data_iter2),
                       [p_data3] "+r" (_p_data_iter3)
                     : [num_t] "r" (NET_L1_WEIGHT_LEN / 4)
                     : "s5", "s6", "s7", "s8", "s9");

        // store the values as 1 byte in the appropriate position
        *(p_result + _ch + 0 * NET_C) = _acc0;
        *(p_result + _ch + 1 * NET_C) = _acc1;
        *(p_result + _ch + 2 * NET_C) = _acc2;
        *(p_result + _ch + 3 * NET_C) = _acc3;
    }
}

/**
 * @brief this function computes the convolution of 4 values in time of all channels at a split transition
 *
 * @param p_data_a pointer to the current data where we start the convolution of split part a
 * @param p_data_b pointer to the current data where we start the convolution of split part b
 * @param stride_a number of elements in a single row of split part a
 * @param stride_b number of elements in a single row of split part b
 * @param num_elems_in_a Number of elements which are in part A. Must be a multiple of 4
 * @param p_weight Pointer to the weight vector
 * @param offset Amount to offset the result at the end of the computation
 * @param p_result pointer to the result data of size [4, NET_C_ALIGN], must be thread local data
 */
void _net_fused_layer_1_2_kernel_conv_transition(const int8_t* p_data_a,
                                                 const int8_t* p_data_b,
                                                 unsigned int stride_a,
                                                 unsigned int stride_b,
                                                 unsigned int num_elems_in_a,
                                                 const int8_t* p_weight,
                                                 uint32_t offset,
                                                 int32_t* p_result) {

    // setup iterators
    const int8_t* _p_data_iter0;
    const int8_t* _p_data_iter1;
    const int8_t* _p_data_iter2;
    const int8_t* _p_data_iter3;
    const int8_t* _p_weight_iter = p_weight;
    int32_t* _p_result_iter = p_result;

    // declare local variables
    int32_t _acc0, _acc1, _acc2, _acc3;

    for (int _ch = 0; _ch < NET_C; _ch++) {

        // setup the iteration
        _p_weight_iter = p_weight;

        _acc0 = offset;
        _acc1 = offset;
        _acc2 = offset;
        _acc3 = offset;

        // part in split A
        _p_data_iter0 = p_data_a + _ch * stride_a;
        _p_data_iter1 = p_data_a + _ch * stride_a + 1 * _T_SPLIT_MEM_SIZE;
        _p_data_iter2 = p_data_a + _ch * stride_a + 2 * _T_SPLIT_MEM_SIZE;
        _p_data_iter3 = p_data_a + _ch * stride_a + 3 * _T_SPLIT_MEM_SIZE;

        asm volatile("lp.setup x0,%[num_t],36;"
                     "   p.lw s9,4(%[p_weight]!);"
                     "   p.lw s5,4(%[p_data0]!);"
                     "   p.lw s6,4(%[p_data1]!);"
                     "   p.lw s7,4(%[p_data2]!);"
                     "   p.lw s8,4(%[p_data3]!);"
                     "   pv.sdotsp.b %[_acc0],s5,s9;"
                     "   pv.sdotsp.b %[_acc1],s6,s9;"
                     "   pv.sdotsp.b %[_acc2],s7,s9;"
                     "36:pv.sdotsp.b %[_acc3],s8,s9;"
                     : [p_weight] "+r" (_p_weight_iter),
                       [_acc0] "+r" (_acc0),
                       [_acc1] "+r" (_acc1),
                       [_acc2] "+r" (_acc2),
                       [_acc3] "+r" (_acc3),
                       [p_data0] "+r" (_p_data_iter0),
                       [p_data1] "+r" (_p_data_iter1),
                       [p_data2] "+r" (_p_data_iter2),
                       [p_data3] "+r" (_p_data_iter3)
                     : [num_t] "r" (num_elems_in_a / 4)
                     : "s5", "s6", "s7", "s8", "s9");

        if (num_elems_in_a < NET_L1_WEIGHT_LEN) {
            // part in split B
            _p_data_iter0 = p_data_b + _ch * stride_b;
            _p_data_iter1 = p_data_b + _ch * stride_b + 1 * _T_SPLIT_MEM_SIZE;
            _p_data_iter2 = p_data_b + _ch * stride_b + 2 * _T_SPLIT_MEM_SIZE;
            _p_data_iter3 = p_data_b + _ch * stride_b + 3 * _T_SPLIT_MEM_SIZE;

            asm volatile("lp.setup x0,%[num_t],36;"
                        "   p.lw s9,4(%[p_weight]!);"
                        "   p.lw s5,4(%[p_data0]!);"
                        "   p.lw s6,4(%[p_data1]!);"
                        "   p.lw s7,4(%[p_data2]!);"
                        "   p.lw s8,4(%[p_data3]!);"
                        "   pv.sdotsp.b %[_acc0],s5,s9;"
                        "   pv.sdotsp.b %[_acc1],s6,s9;"
                        "   pv.sdotsp.b %[_acc2],s7,s9;"
                        "36:pv.sdotsp.b %[_acc3],s8,s9;"
                        : [p_weight] "+r" (_p_weight_iter),
                        [_acc0] "+r" (_acc0),
                        [_acc1] "+r" (_acc1),
                        [_acc2] "+r" (_acc2),
                        [_acc3] "+r" (_acc3),
                        [p_data0] "+r" (_p_data_iter0),
                        [p_data1] "+r" (_p_data_iter1),
                        [p_data2] "+r" (_p_data_iter2),
                        [p_data3] "+r" (_p_data_iter3)
                        : [num_t] "r" ((NET_L1_WEIGHT_LEN - num_elems_in_a) / 4)
                        : "s5", "s6", "s7", "s8", "s9");
        }

        // store the values as 1 byte in the appropriate position
        *(_p_result_iter + 0 * NET_C) = _acc0;
        *(_p_result_iter + 1 * NET_C) = _acc1;
        *(_p_result_iter + 2 * NET_C) = _acc2;
        *(_p_result_iter + 3 * NET_C) = _acc3;

        // go to the next value in the thread data
        _p_result_iter++;
    }
}

/**
 * @brief Compute the result of the dot product for the second layer and add them to the current pooling sum
 *
 * @param p_data Pointer to input data of shape [4, NET_C_ALIGN], must be the thread local data, the result of the function above
 * @param p_weight Pointer to the weight vector of the layer 2
 * @param threshold_0 Threshold for ReLU of the first output channel
 * @param threshold_1 Threshold for ReLU of the second output channel
 * @param p_pool_sum_0 Pointer to the first pool sum value, which is updated in this function
 * @param p_pool_sum_1 Pointer to the second pool sum value, which is updated in this function
 */
void _net_fused_layer_1_2_kernel_dotp_acc(const int32_t* p_data,
                                          const int32_t* p_weight,
                                          int32_t threshold_0,
                                          int32_t threshold_1,
                                          int32_t* p_pool_sum_0,
                                          int32_t* p_pool_sum_1) {

    // iterators
    const int32_t* _p_data_iter = p_data;
    const int32_t* _p_weight_iter = p_weight;

    // local registers
    int32_t _a0, _a1, _a2, _a3;
    int32_t _b0, _b1;
    int32_t _elem_0_0 = 0, _elem_0_1 = 0, _elem_0_2 = 0, _elem_0_3 = 0;
    int32_t _elem_1_0 = 0, _elem_1_1 = 0, _elem_1_2 = 0, _elem_1_3 = 0;

    // registers for the pool sum
    int32_t _pool_sum_0 = *p_pool_sum_0;
    int32_t _pool_sum_1 = *p_pool_sum_1;

    for (int _ch = 0; _ch < NET_C; _ch++) {

        _a0 = *(_p_data_iter + 0 * NET_C);
        _a1 = *(_p_data_iter + 1 * NET_C);
        _a2 = *(_p_data_iter + 2 * NET_C);
        _a3 = *(_p_data_iter + 3 * NET_C);

        _b0 = *_p_weight_iter;
        _b1 = *(_p_weight_iter + NET_L2_WEIGHT_LEN);

        _p_data_iter++;
        _p_weight_iter++;

        _elem_0_0 = __MAC(_elem_0_0, _b0, _a0);
        _elem_0_1 = __MAC(_elem_0_1, _b0, _a1);
        _elem_0_2 = __MAC(_elem_0_2, _b0, _a2);
        _elem_0_3 = __MAC(_elem_0_3, _b0, _a3);

        _elem_1_0 = __MAC(_elem_1_0, _b1, _a0);
        _elem_1_1 = __MAC(_elem_1_1, _b1, _a1);
        _elem_1_2 = __MAC(_elem_1_2, _b1, _a2);
        _elem_1_3 = __MAC(_elem_1_3, _b1, _a3);

    }

    // do ReLU on the first and second element
    _elem_0_0 = __MAX(_elem_0_0, threshold_0);
    _elem_0_1 = __MAX(_elem_0_1, threshold_0);
    _elem_0_2 = __MAX(_elem_0_2, threshold_0);
    _elem_0_3 = __MAX(_elem_0_3, threshold_0);

    _elem_1_0 = __MAX(_elem_1_0, threshold_1);
    _elem_1_1 = __MAX(_elem_1_1, threshold_1);
    _elem_1_2 = __MAX(_elem_1_2, threshold_1);
    _elem_1_3 = __MAX(_elem_1_3, threshold_1);

    // sum up the values
    _pool_sum_0 += (_elem_0_0 + _elem_0_1) + (_elem_0_2 + _elem_0_3);
    _pool_sum_1 += (_elem_1_0 + _elem_1_1) + (_elem_1_2 + _elem_1_3);

    // store the results back
    *p_pool_sum_0 = _pool_sum_0;
    *p_pool_sum_1 = _pool_sum_1;
}

/**
 * @brief Applies the final transformation of the pooling result and stores it back into the array
 *
 * @param pool_sum_0 Sum of all results for output cannel 0
 * @param pool_sum_1 Sum of all results for output cannel 1
 * @param factor_0 Scaling division factor for output cannel 0
 * @param factor_1 Scaling division factor for output cannel 1
 * @param offset_0 Offset for output channel 0
 * @param offset_1 Offset for output channel 1
 * @param p_result Pointer to result array (already at the correct position)
 */
inline void _net_fused_layer_1_2_kernel_store_result(int32_t pool_sum_0,
                                                     int32_t pool_sum_1,
                                                     int32_t factor_0,
                                                     int32_t factor_1,
                                                     int32_t offset_0,
                                                     int32_t offset_1,
                                                     int8_t* p_result) {

    // scale it
    pool_sum_0 = (pool_sum_0 + offset_0) / factor_0;
    pool_sum_1 = (pool_sum_1 + offset_1) / factor_1;

    // clip it
    pool_sum_0 = __CLIP_R(pool_sum_0, 127);
    pool_sum_1 = __CLIP_R(pool_sum_1, 127);

    // store it
    *(p_result + 0 * NET_T8_ALIGN) = pool_sum_0;
    *(p_result + 1 * NET_T8_ALIGN) = pool_sum_1;

}

typedef struct {
    const int8_t* p_data_ext;
    int8_t* p_data;
    int8_t* p_result;

    int8_t* p_weight_l1;
    int32_t* p_factor_l1;
    int32_t* p_offset_l1;

    int32_t* p_weight_l2;
    int32_t* p_factor_l2;
    int32_t* p_offset_l2;

    int32_t* p_thread_data;
} _net_fused_layer_1_2_kernel_t;

/**
 * @brief Kernel for doing the computation
 */
void _net_fused_layer_1_2_kernel(void* args) {

    unsigned int _core_id = rt_core_id();

    // get values from args
    _net_fused_layer_1_2_kernel_t* _args = args;

    const int8_t* _p_data_ext = _args->p_data_ext;
    int8_t* _p_data_a = _args->p_data;
    int8_t* _p_result = _args->p_result;
    int8_t* _p_weight_l1 = _args->p_weight_l1;
    int32_t* _p_factor_l1 = _args->p_factor_l1;
    int32_t* _p_offset_l1 = _args->p_offset_l1;
    int32_t* _p_weight_l2 = _args->p_weight_l2;
    int32_t* _p_factor_l2 = _args->p_factor_l2;
    int32_t* _p_offset_l2 = _args->p_offset_l2;
    int32_t* _p_thread_data = _args->p_thread_data;

    int8_t* _p_data_b = _p_data_a + 4 * _T_SPLIT_MEM_SIZE;

    // change the pointers to point to the data used by the specific core
    _p_result += _core_id * 2 * NET_T8_ALIGN;
    _p_weight_l1 += _core_id * NET_L1_WEIGHT_LEN_ALIGN;
    _p_factor_l1 += _core_id;
    _p_offset_l1 += _core_id;
    _p_weight_l2 += _core_id * 2 * NET_L2_WEIGHT_LEN;
    _p_factor_l2 += _core_id * 2;
    _p_offset_l2 += _core_id * 2;
    _p_thread_data += _core_id * (NET_C * 4 + _THREAD_MEM_OFFSET);

    // load the scaling factors
    int32_t _factor_l1 = *_p_factor_l1;
    int32_t _offset_l1 = *_p_offset_l1;
    int32_t _factor_l2_0 = *(_p_factor_l2 + 0) * _factor_l1;
    int32_t _offset_l2_0 = *(_p_offset_l2 + 0) * _factor_l1;
    int32_t _factor_l2_1 = *(_p_factor_l2 + 1) * _factor_l1;
    int32_t _offset_l2_1 = *(_p_offset_l2 + 1) * _factor_l1;

    // compute the ReLU threshold
    int32_t _threshold_0 = -(_offset_l2_0 >> 3);
    int32_t _threshold_1 = -(_offset_l2_1 >> 3);

    int8_t* _p_data_iter;            // iterator over the current elements for which we do the computation
    int8_t* _p_data_iter2;           // iterator over the current elements for which we do the computation, for the reference to array 2
    int8_t* _p_result_iter = _p_result;

    // variable to count down from NET_L1_WEIGHT_LEN to see how many samples are in the first range and how many in the second.
    int _num_comp_in_range_1;

    // registers for the second layer
    int32_t _pool_sum_0;
    int32_t _pool_sum_1;

    // copy the first data over
    rt_dma_copy_t _copy_start;
    rt_dma_copy_t _copy_comp;

    // Copy the first split and also start copying the second split over
    if (_core_id == 0) {
        rt_dma_memcpy_2d((unsigned int)_p_data_ext,
                         (unsigned int)_p_data_a + 0 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 0, &_copy_start);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 1,
                         (unsigned int)_p_data_a + 1 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_start);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 2,
                         (unsigned int)_p_data_a + 2 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_start);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 3,
                         (unsigned int)_p_data_a + 3 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_start);

        // also start to copy the next part over
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + _T_SPLIT_LEN,
                         (unsigned int)_p_data_b + 0 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 0, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + _T_SPLIT_LEN + 1,
                         (unsigned int)_p_data_b + 1 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + _T_SPLIT_LEN + 2,
                         (unsigned int)_p_data_b + 2 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + _T_SPLIT_LEN + 3,
                         (unsigned int)_p_data_b + 3 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);

        // wait for the start dma to finish
        rt_dma_wait(&_copy_start);
    }

    rt_team_barrier();

    /***********
     * Region 1: 0 .. _T_SPLIT_LEN - L1_WEIGHT_LEN
     ***********/

    _p_data_iter = _p_data_a;

    for (int _t_out = 0; _t_out < (_T_SPLIT_LEN - NET_L1_WEIGHT_LEN) / 8; _t_out++) {

        // reset the pooling summation register
        _pool_sum_0 = 0;
        _pool_sum_1 = 0;

        // iterate over all the padding samples divided by 4, because we compute 4 values at the same time
        for (int _t_pad = 0; _t_pad < 8 / 4; _t_pad++) {
            // compute the intermediate vector
            _net_fused_layer_1_2_kernel_conv(_core_id, _p_data_iter, _T_SPLIT_LEN, _p_weight_l1, _offset_l1, _p_thread_data);

            // move to the next 4 time samples
            _p_data_iter += 4;

            // compute the dot product of the layer 2, and add accumulate the values for padding.
            _net_fused_layer_1_2_kernel_dotp_acc(_p_thread_data, _p_weight_l2, _threshold_0, _threshold_1, &_pool_sum_0, &_pool_sum_1);
        }

        // transform it and store back to memory
        _net_fused_layer_1_2_kernel_store_result(_pool_sum_0, _pool_sum_1, _factor_l2_0, _factor_l2_1, _offset_l2_0, _offset_l2_1, _p_result_iter++);
    }

    /***********
     * Region 2: _T_SPLIT_LEN - L1_WEIGHT_LEN .. T_SPLIT_LEN
     ***********/

    // wait until the second part of the data is copied into slot B
    if (_core_id == 0) {
        rt_dma_wait(&_copy_comp);
    }
    rt_team_barrier();

    _p_data_iter = _p_data_a + (_T_SPLIT_LEN - NET_L1_WEIGHT_LEN);

    // this counter is counted down by 1 after every dot product computation, to use more and more of the new data.
    _num_comp_in_range_1 = NET_L1_WEIGHT_LEN;

    for (int _t_out = 0; _t_out < NET_L1_WEIGHT_LEN / 8; _t_out++) {

        // reset the pooling summation register
        _pool_sum_0 = 0;
        _pool_sum_1 = 0;

        // iterate over all the padding samples divided by 4, because we compute 4 values at the same time
        for (int _t_pad = 0; _t_pad < 8 / 4; _t_pad++) {

            // compute the intermediate vector
            _net_fused_layer_1_2_kernel_conv_transition(_p_data_iter, _p_data_b,
                                                        _T_SPLIT_LEN, _T_SPLIT_LEN,
                                                        _num_comp_in_range_1, _p_weight_l1, _offset_l1,
                                                        _p_thread_data);

            // move to the next 4 time samples, the second iterator does not need to be updated
            _p_data_iter += 4;

            // decrement the num_comp_in_range_a counter to make sure that we use less of range a and more in range b.
            _num_comp_in_range_1 -= 4;

            // compute the dot product of the layer 2, and add accumulate the values for padding.
            _net_fused_layer_1_2_kernel_dotp_acc(_p_thread_data, _p_weight_l2, _threshold_0, _threshold_1, &_pool_sum_0, &_pool_sum_1);
        }

        // transform it and store back to memory
        _net_fused_layer_1_2_kernel_store_result(_pool_sum_0, _pool_sum_1, _factor_l2_0, _factor_l2_1, _offset_l2_0, _offset_l2_1, _p_result_iter++);
    }

    rt_team_barrier();

    /***********
     * Region 3: T_SPLIT_LEN .. 2 * T_SPLIT_LEN - NET_L1_WEIGHT_LEN
     ***********/

    // data in slot A is no longer used! copy the data of split 3 over to slot A
    if (_core_id == 0) {
        // also start to copy the next part over
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 2 *_T_SPLIT_LEN,
                         (unsigned int)_p_data_a + 0 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 0, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 2 *_T_SPLIT_LEN + 1,
                         (unsigned int)_p_data_a + 1 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 2 *_T_SPLIT_LEN + 2,
                         (unsigned int)_p_data_a + 2 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 2 *_T_SPLIT_LEN + 3,
                         (unsigned int)_p_data_a + 3 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
    }

    _p_data_iter = _p_data_b;

    for (int _t_out = 0; _t_out < (_T_SPLIT_LEN - NET_L1_WEIGHT_LEN) / 8; _t_out++) {

        // reset the pooling summation register
        _pool_sum_0 = 0;
        _pool_sum_1 = 0;

        // iterate over all the padding samples divided by 4, because we compute 4 values at the same time
        for (int _t_pad = 0; _t_pad < 8 / 4; _t_pad++) {
            // compute the intermediate vector
            _net_fused_layer_1_2_kernel_conv(_core_id, _p_data_iter, _T_SPLIT_LEN, _p_weight_l1, _offset_l1, _p_thread_data);

            // move to the next 4 time samples
            _p_data_iter += 4;

            // compute the dot product of the layer 2, and add accumulate the values for padding.
            _net_fused_layer_1_2_kernel_dotp_acc(_p_thread_data, _p_weight_l2, _threshold_0, _threshold_1, &_pool_sum_0, &_pool_sum_1);
        }

        // transform it and store back to memory
        _net_fused_layer_1_2_kernel_store_result(_pool_sum_0, _pool_sum_1, _factor_l2_0, _factor_l2_1, _offset_l2_0, _offset_l2_1, _p_result_iter++);
    }

    /***********
     * Region 4: 2 * _T_SPLIT_LEN - NET_L1_WEIGHT_LEN .. 2 * T_SPLIT_LEN
     ***********/

    // wait until the second part of the data is copied into slot A
    if (_core_id == 0) {
        rt_dma_wait(&_copy_comp);
    }
    rt_team_barrier();

    _p_data_iter = _p_data_b + (_T_SPLIT_LEN - NET_L1_WEIGHT_LEN);

    // this counter is counted down by 1 after every dot product computation, to use more and more of the new data.
    _num_comp_in_range_1 = NET_L1_WEIGHT_LEN;

    for (int _t_out = 0; _t_out < NET_L1_WEIGHT_LEN / 8; _t_out++) {

        // reset the pooling summation register
        _pool_sum_0 = 0;
        _pool_sum_1 = 0;

        // iterate over all the padding samples divided by 4, because we compute 4 values at the same time
        for (int _t_pad = 0; _t_pad < 8 / 4; _t_pad++) {

            // compute the intermediate vector
            _net_fused_layer_1_2_kernel_conv_transition(_p_data_iter, _p_data_a,
                                                        _T_SPLIT_LEN, _T_SPLIT_LEN,
                                                        _num_comp_in_range_1, _p_weight_l1, _offset_l1,
                                                        _p_thread_data);

            // move to the next 4 time samples, the second iterator does not need to be updated
            _p_data_iter += 4;

            // decrement the num_comp_in_range_a counter to make sure that we use less of range a and more in range b.
            _num_comp_in_range_1 -= 4;

            // compute the dot product of the layer 2, and add accumulate the values for padding.
            _net_fused_layer_1_2_kernel_dotp_acc(_p_thread_data, _p_weight_l2, _threshold_0, _threshold_1, &_pool_sum_0, &_pool_sum_1);
        }

        // transform it and store back to memory
        _net_fused_layer_1_2_kernel_store_result(_pool_sum_0, _pool_sum_1, _factor_l2_0, _factor_l2_1, _offset_l2_0, _offset_l2_1, _p_result_iter++);
    }

    rt_team_barrier();

    /***********
     * Region 5: 2 * T_SPLIT_LEN .. 3 * T_SPLIT_LEN - NET_L1_WEIGHT_LEN
     ***********/

    // data in slot b is no longer used! copy the data of split 4 over to slot b
    if (_core_id == 0) {
        // also start to copy the next part over
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 3 *_T_SPLIT_LEN,
                         (unsigned int)_p_data_b + 0 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 0, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 3 *_T_SPLIT_LEN + 1,
                         (unsigned int)_p_data_b + 1 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 3 *_T_SPLIT_LEN + 2,
                         (unsigned int)_p_data_b + 2 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 3 *_T_SPLIT_LEN + 3,
                         (unsigned int)_p_data_b + 3 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN, // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
    }

    _p_data_iter = _p_data_a;

    for (int _t_out = 0; _t_out < (_T_SPLIT_LEN - NET_L1_WEIGHT_LEN) / 8; _t_out++) {

        // reset the pooling summation register
        _pool_sum_0 = 0;
        _pool_sum_1 = 0;

        // iterate over all the padding samples divided by 4, because we compute 4 values at the same time
        for (int _t_pad = 0; _t_pad < 8 / 4; _t_pad++) {
            // compute the intermediate vector
            _net_fused_layer_1_2_kernel_conv(_core_id, _p_data_iter, _T_SPLIT_LEN, _p_weight_l1, _offset_l1, _p_thread_data);

            // move to the next 4 time samples
            _p_data_iter += 4;

            // compute the dot product of the layer 2, and add accumulate the values for padding.
            _net_fused_layer_1_2_kernel_dotp_acc(_p_thread_data, _p_weight_l2, _threshold_0, _threshold_1, &_pool_sum_0, &_pool_sum_1);
        }

        // transform it and store back to memory
        _net_fused_layer_1_2_kernel_store_result(_pool_sum_0, _pool_sum_1, _factor_l2_0, _factor_l2_1, _offset_l2_0, _offset_l2_1, _p_result_iter++);
    }

    /***********
     * Region 6: 3 * _T_SPLIT_LEN - NET_L1_WEIGHT_LEN .. 3 * T_SPLIT_LEN
     ***********/

    // wait until the second part of the data is copied into slot B
    if (_core_id == 0) {
        rt_dma_wait(&_copy_comp);
    }
    rt_team_barrier();

    _p_data_iter = _p_data_a + (_T_SPLIT_LEN - NET_L1_WEIGHT_LEN);

    // this counter is counted down by 1 after every dot product computation, to use more and more of the new data.
    _num_comp_in_range_1 = NET_L1_WEIGHT_LEN;

    for (int _t_out = 0; _t_out < NET_L1_WEIGHT_LEN / 8; _t_out++) {

        // reset the pooling summation register
        _pool_sum_0 = 0;
        _pool_sum_1 = 0;

        // iterate over all the padding samples divided by 4, because we compute 4 values at the same time
        for (int _t_pad = 0; _t_pad < 8 / 4; _t_pad++) {

            // compute the intermediate vector
            _net_fused_layer_1_2_kernel_conv_transition(_p_data_iter, _p_data_b,
                                                        _T_SPLIT_LEN, _T_SPLIT_LEN,
                                                        _num_comp_in_range_1, _p_weight_l1, _offset_l1,
                                                        _p_thread_data);

            // move to the next 4 time samples, the second iterator does not need to be updated
            _p_data_iter += 4;

            // decrement the num_comp_in_range_a counter to make sure that we use less of range a and more in range b.
            _num_comp_in_range_1 -= 4;

            // compute the dot product of the layer 2, and add accumulate the values for padding.
            _net_fused_layer_1_2_kernel_dotp_acc(_p_thread_data, _p_weight_l2, _threshold_0, _threshold_1, &_pool_sum_0, &_pool_sum_1);
        }

        // transform it and store back to memory
        _net_fused_layer_1_2_kernel_store_result(_pool_sum_0, _pool_sum_1, _factor_l2_0, _factor_l2_1, _offset_l2_0, _offset_l2_1, _p_result_iter++);
    }

    rt_team_barrier();

    /***********
     * Region 7: 3 * T_SPLIT_LEN .. 4 * T_SPLIT_LEN - NET_L1_WEIGHT_LEN
     ***********/

    // data in slot A is no longer used! copy the data of split 5 (different size) over to slot A
    if (_core_id == 0) {
        // also start to copy the next part over
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 4 *_T_SPLIT_LEN,
                         (unsigned int)_p_data_a + 0 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN_LAST, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN,      // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN_LAST,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 0, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 4 *_T_SPLIT_LEN + 1,
                         (unsigned int)_p_data_a + 1 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN_LAST, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN,      // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN_LAST,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 4 *_T_SPLIT_LEN + 2,
                         (unsigned int)_p_data_a + 2 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN_LAST, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN,      // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN_LAST,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
        rt_dma_memcpy_2d((unsigned int)_p_data_ext + 4 *_T_SPLIT_LEN + 3,
                         (unsigned int)_p_data_a + 3 * _T_SPLIT_MEM_SIZE,
                         sizeof(int8_t) * NET_C * _T_SPLIT_LEN_LAST, // number of elements in total
                         sizeof(int8_t) * NET_L1_PAD_INPUT_LEN,      // length of each line (row)
                         sizeof(int8_t) * _T_SPLIT_LEN_LAST,         // number of elements to transfer per line
                         RT_DMA_DIR_EXT2LOC, 1, &_copy_comp);
    }

    _p_data_iter = _p_data_b;

    for (int _t_out = 0; _t_out < (_T_SPLIT_LEN - NET_L1_WEIGHT_LEN) / 8; _t_out++) {

        // reset the pooling summation register
        _pool_sum_0 = 0;
        _pool_sum_1 = 0;

        // iterate over all the padding samples divided by 4, because we compute 4 values at the same time
        for (int _t_pad = 0; _t_pad < 8 / 4; _t_pad++) {
            // compute the intermediate vector
            _net_fused_layer_1_2_kernel_conv(_core_id, _p_data_iter, _T_SPLIT_LEN, _p_weight_l1, _offset_l1, _p_thread_data);

            // move to the next 4 time samples
            _p_data_iter += 4;

            // compute the dot product of the layer 2, and add accumulate the values for padding.
            _net_fused_layer_1_2_kernel_dotp_acc(_p_thread_data, _p_weight_l2, _threshold_0, _threshold_1, &_pool_sum_0, &_pool_sum_1);
        }

        // transform it and store back to memory
        _net_fused_layer_1_2_kernel_store_result(_pool_sum_0, _pool_sum_1, _factor_l2_0, _factor_l2_1, _offset_l2_0, _offset_l2_1, _p_result_iter++);
    }

    /***********
     * Region 8: 4 * _T_SPLIT_LEN - L1_WEIGHT_LEN .. 4 * T_SPLIT_LEN
     ***********/

    // wait until the second part of the data is copied into slot A
    if (_core_id == 0) {
        rt_dma_wait(&_copy_comp);
    }
    rt_team_barrier();

    _p_data_iter = _p_data_b + (_T_SPLIT_LEN - NET_L1_WEIGHT_LEN);

    // this counter is counted down by 1 after every dot product computation, to use more and more of the new data.
    _num_comp_in_range_1 = NET_L1_WEIGHT_LEN;

    for (int _t_out = 0; _t_out < NET_L1_WEIGHT_LEN / 8; _t_out++) {

        // reset the pooling summation register
        _pool_sum_0 = 0;
        _pool_sum_1 = 0;

        // iterate over all the padding samples divided by 4, because we compute 4 values at the same time
        for (int _t_pad = 0; _t_pad < 8 / 4; _t_pad++) {

            // compute the intermediate vector
            _net_fused_layer_1_2_kernel_conv_transition(_p_data_iter, _p_data_a,
                                                        _T_SPLIT_LEN, _T_SPLIT_LEN_LAST,
                                                        _num_comp_in_range_1, _p_weight_l1, _offset_l1,
                                                        _p_thread_data);

            // move to the next 4 time samples, the second iterator does not need to be updated
            _p_data_iter += 4;

            // decrement the num_comp_in_range_a counter to make sure that we use less of range a and more in range b.
            _num_comp_in_range_1 -= 4;

            // compute the dot product of the layer 2, and add accumulate the values for padding.
            _net_fused_layer_1_2_kernel_dotp_acc(_p_thread_data, _p_weight_l2, _threshold_0, _threshold_1, &_pool_sum_0, &_pool_sum_1);
        }

        // transform it and store back to memory
        _net_fused_layer_1_2_kernel_store_result(_pool_sum_0, _pool_sum_1, _factor_l2_0, _factor_l2_1, _offset_l2_0, _offset_l2_1, _p_result_iter++);
    }

    rt_team_barrier();

    /***********
     * Region 9: 4 * T_SPLIT_LEN .. 4 * T_SPLIT_LEN + _T_SPLIT_LEN_LAST
     ***********/

    _p_data_iter = _p_data_a;

    for (int _t_out = 0; _t_out < (_T_SPLIT_LEN_LAST - NET_L1_WEIGHT_LEN) / 8; _t_out++) {

        // reset the pooling summation register
        _pool_sum_0 = 0;
        _pool_sum_1 = 0;

        // iterate over all the padding samples divided by 4, because we compute 4 values at the same time
        for (int _t_pad = 0; _t_pad < 8 / 4; _t_pad++) {
            // compute the intermediate vector
            _net_fused_layer_1_2_kernel_conv(_core_id, _p_data_iter, _T_SPLIT_LEN_LAST, _p_weight_l1, _offset_l1, _p_thread_data);

            // move to the next 4 time samples
            _p_data_iter += 4;

            // compute the dot product of the layer 2, and add accumulate the values for padding.
            _net_fused_layer_1_2_kernel_dotp_acc(_p_thread_data, _p_weight_l2, _threshold_0, _threshold_1, &_pool_sum_0, &_pool_sum_1);
        }

        // transform it and store back to memory
        _net_fused_layer_1_2_kernel_store_result(_pool_sum_0, _pool_sum_1, _factor_l2_0, _factor_l2_1, _offset_l2_0, _offset_l2_1, _p_result_iter++);
    }

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
    int8_t* _p_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * 8 * _T_SPLIT_MEM_SIZE);

    int8_t* _p_result_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    int8_t* _p_weight_l1_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * NET_F1 * NET_L1_WEIGHT_LEN_ALIGN);
    int32_t* _p_factor_l1_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F1);
    int32_t* _p_offset_l1_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F1);

    int32_t* _p_weight_l2_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    int32_t* _p_factor_l2_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_l2_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    int32_t* _p_thread_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NUM_WORKERS * (NET_C * 4 + _THREAD_MEM_OFFSET));

    // error handling
    if (_p_thread_data_loc == NULL) {
        printf("Error! Not enough space in L1 memory!");
        return;
    }

    rt_dma_copy_t _copy;

    // load all the weights of layer 1
    rt_dma_memcpy((unsigned int)net_l1_weight_reverse_pad,
                  (unsigned int)_p_weight_l1_loc,
                  sizeof(int8_t) * NET_F1 * NET_L1_WEIGHT_LEN_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_factor,
                  (unsigned int)_p_factor_l1_loc,
                  sizeof(int32_t) * NET_F1,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);
    rt_dma_memcpy((unsigned int)net_l1_offset,
                  (unsigned int)_p_offset_l1_loc,
                  sizeof(int32_t) * NET_F1,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    // load all the weights of layer 2
    rt_dma_memcpy((unsigned int)net_l2_weight_32,
                  (unsigned int)_p_weight_l2_loc,
                  sizeof(int32_t) * NET_F2 * NET_L2_WEIGHT_LEN,
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
    _args.p_data_ext = p_data;
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
    rt_free(RT_ALLOC_CL_DATA, _p_data_loc, sizeof(int8_t) * 8 * _T_SPLIT_MEM_SIZE);
    rt_free(RT_ALLOC_CL_DATA, _p_result_loc, sizeof(int8_t) * NET_F2 * NET_T8_ALIGN);

    rt_free(RT_ALLOC_CL_DATA, _p_weight_l1_loc, sizeof(int8_t) * NET_F1 * NET_L1_WEIGHT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_l1_loc, sizeof(int32_t) * NET_F1);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_l1_loc, sizeof(int32_t) * NET_F1);

    rt_free(RT_ALLOC_CL_DATA, _p_weight_l2_loc, sizeof(int8_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    rt_free(RT_ALLOC_CL_DATA, _p_factor_l2_loc, sizeof(int32_t) * NET_F2);
    rt_free(RT_ALLOC_CL_DATA, _p_offset_l2_loc, sizeof(int32_t) * NET_F2);

    rt_free(RT_ALLOC_CL_DATA, _p_thread_data_loc, sizeof(int32_t) * NUM_WORKERS * (NET_C * 4 + _THREAD_MEM_OFFSET));

}


#else //DUPLICATE_FEATUREMAP

typedef struct {
    int8_t* p_data;
    int8_t* p_result;

    int8_t* p_weight_l1;
    int32_t* p_factor_l1;
    int32_t* p_offset_l1;

    int32_t* p_weight_l2;
    int32_t* p_factor_l2;
    int32_t* p_offset_l2;

    int32_t* p_thread_data;
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
    int32_t* _p_weight_l2 = _args->p_weight_l2;
    int32_t* _p_factor_l2 = _args->p_factor_l2;
    int32_t* _p_offset_l2 = _args->p_offset_l2;
    int32_t* _p_thread_data = _args->p_thread_data;

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
    int32_t _factor_l2_0 = *(_p_factor_l2 + 0) * _factor_l1;
    int32_t _offset_l2_0 = *(_p_offset_l2 + 0) * _factor_l1;
    int32_t _factor_l2_1 = *(_p_factor_l2 + 1) * _factor_l1;
    int32_t _offset_l2_1 = *(_p_offset_l2 + 1) * _factor_l1;

    // compute the ReLU threshold
    int32_t _threshold_0 = -(_offset_l2_0 >> 3);
    int32_t _threshold_1 = -(_offset_l2_1 >> 3);

    int8_t* _p_data_iter = _p_data; // iterator over the current elements for which we do the computation
    int8_t* _p_data_iter_comp;      // Pointer to the data while doing the dot product
    int8_t* _p_weight_l1_iter_comp; // pointer to the weights while doing the dot product
    int32_t* _p_thread_data_iter;    // iterator over the thread data
    int8_t* _p_result_iter = _p_result;
    int32_t* _p_weight_l2_iter_comp;

    // registers for the first layer
    v4s _x0, _x1, _x2, _x3;
    v4s _y;
    int32_t _acc0, _acc1, _acc2, _acc3;

    // registers for the second layer
    int32_t _pool_sum_0;
    int32_t _pool_sum_1;
    int32_t _elem_0, _elem_1;
    int32_t _a, _b0, _b1;

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
                _acc0 = _acc0 + _offset_l1;
                _acc1 = _acc1 + _offset_l1;
                _acc2 = _acc2 + _offset_l1;
                _acc3 = _acc3 + _offset_l1;

                // store the values as 1 byte in the appropriate position
                *(_p_thread_data_iter + 0 * NET_C_ALIGN) = _acc0;
                *(_p_thread_data_iter + 1 * NET_C_ALIGN) = _acc1;
                *(_p_thread_data_iter + 2 * NET_C_ALIGN) = _acc2;
                *(_p_thread_data_iter + 3 * NET_C_ALIGN) = _acc3;

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

                _p_thread_data_iter = _p_thread_data + _i * NET_C_ALIGN;
                _p_weight_l2_iter_comp = _p_weight_l2;
                _elem_0 = 0;
                _elem_1 = 0;

                for (int _ch = 0; _ch < NET_C; _ch++) {
                    _a = *_p_thread_data_iter;
                    _b0 = *_p_weight_l2_iter_comp;
                    _b1 = *(_p_weight_l2_iter_comp + NET_L2_WEIGHT_LEN);

                    _elem_0 = __MAC(_elem_0, _a, _b0);
                    _elem_1 = __MAC(_elem_1, _a, _b1);

                    _p_thread_data_iter++;
                    _p_weight_l2_iter_comp++;
                }

                // do ReLU on the first and second element
                _elem_0 = __MAX(_elem_0, _threshold_0);
                _elem_1 = __MAX(_elem_1, _threshold_1);

                // add them to the pooling sum
                _pool_sum_0 += _elem_0;
                _pool_sum_1 += _elem_1;
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

    int32_t* _p_weight_l2_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2 * NET_L2_WEIGHT_LEN);
    int32_t* _p_factor_l2_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);
    int32_t* _p_offset_l2_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NET_F2);

    int32_t* _p_thread_data_loc = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * NUM_WORKERS * NET_C_ALIGN * 4);

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
    rt_dma_memcpy((unsigned int)net_l2_weight_32,
                  (unsigned int)_p_weight_l2_loc,
                  sizeof(int32_t) * NET_F2 * NET_L2_WEIGHT_LEN,
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

    rt_free(RT_ALLOC_CL_DATA, _p_thread_data_loc, sizeof(int32_t) * NUM_WORKERS * NET_C_ALIGN * 4);

}

#endif//DUPLICATE_FEATUREMAP

#else //NO_INTERMEDIATE_SCALE

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

    // compute the ReLU threshold
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
                _acc0 = _acc0 + _offset_l1;
                _acc1 = _acc1 + _offset_l1;
                _acc2 = _acc2 + _offset_l1;
                _acc3 = _acc3 + _offset_l1;

                _acc0 = _acc0 / _factor_l1;
                _acc1 = _acc1 / _factor_l1;
                _acc2 = _acc2 / _factor_l1;
                _acc3 = _acc3 / _factor_l1;

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

#endif //NO_INTERMEDIATE_SCALE

#endif //FUSE_LAYERS
