/**
 * @file conv.h
 * @author Tibor Schneider
 * @date 2020/01/25
 * @brief Implementation of transformation from 32bit to 8bit
 */

#include "rt/rt_api.h"
#include "functional.h"
#include "stdio.h"

#define SIGN_BIT(x) __BITEXTRACTU(x, 1, 31)

/**
 * @brief Convert a vector of 32bits back to 8bit (by scaling and shifting)
 *
 * Per element k, p_res[k] = (p_in[k * stride] + div_factor / 2) / div_factor
 * 
 * @warning Data must be already present in L1 memory, and the output vector must 
 * be allocated
 *
 * @param p_in Pointer to input vector a on L1 memory
 * @param len Length of the input and output vector
 * @param div_factor factor by which to divide
 * @param stride Collect elements from p_in with distance stride apart. Set to 1 for default stride
 * @param p_res Pointer to the output vector.
 */
void func_transform_32to8(const int32_t* p_in,
                          unsigned int len,
                          int32_t div_factor,
                          unsigned int stride,
                          int8_t* p_res) {

    int32_t _a, _b, _c, _d;             // temporary values
    const int32_t* _p_x = p_in;      // pointer to current element in x

#ifdef ROUND
    uint32_t _sa, _sb, _sc, _sd;                          // sign bit of a, b, c and d
    int32_t _offset = div_factor / 2;                  // offset is added every time
    int32_t _neg_offset = 1 - ((div_factor / 2) * 2);  // neg offset is only added when the number if negative
#endif //ROUND

    unsigned int _num_blk = len / 4;
    unsigned int _num_rem = len % 4;

    // do the elements which can be unrolled
    while (_num_blk > 0) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);
        _c = *(_p_x + 2 * stride);
        _d = *(_p_x + 3 * stride);

        _p_x += 4 * stride;

#ifdef ROUND
        _sa = SIGN_BIT(_a);
        _a += _offset;
        _a = __MAC(_a, _sa, _neg_offset);

        _sb = SIGN_BIT(_b);
        _b += _offset;
        _b = __MAC(_b, _sb, _neg_offset);

        _sc = SIGN_BIT(_c);
        _c += _offset;
        _c = __MAC(_c, _sc, _neg_offset);

        _sd = SIGN_BIT(_d);
        _d += _offset;
        _d = __MAC(_d, _sd, _neg_offset);
#endif //ROUND

        _a = _a / div_factor;
        _b = _b / div_factor;
        _c = _c / div_factor;
        _d = _d / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);
        _c = __CLIP_R(_c, 127);
        _d = __CLIP_R(_d, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, _c, _d);
        
        p_res += 4;
        _num_blk--;
    }

    if (_num_rem == 1) {

        _a = *_p_x;

#ifdef ROUND
        _sa = SIGN_BIT(_a);
        _a += _offset;
        _a = __MAC(_a, _sa, _neg_offset);
#endif //ROUND

        _a = _a / div_factor;

        _a = __CLIP_R(_a, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, 0, 0, 0);

    } else if (_num_rem == 2) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);

#ifdef ROUND
        _sa = SIGN_BIT(_a);
        _a += _offset;
        _a = __MAC(_a, _sa, _neg_offset);

        _sb = SIGN_BIT(_b);
        _b += _offset;
        _b = __MAC(_b, _sb, _neg_offset);
#endif //ROUND

        _a = _a / div_factor;
        _b = _b / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, 0, 0);

    } else if (_num_rem == 3) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);
        _c = *(_p_x + 2 * stride);

#ifdef ROUND
        _sa = SIGN_BIT(_a);
        _a += _offset;
        _a = __MAC(_a, _sa, _neg_offset);

        _sb = SIGN_BIT(_b);
        _b += _offset;
        _b = __MAC(_b, _sb, _neg_offset);

        _sc = SIGN_BIT(_c);
        _c += _offset;
        _c = __MAC(_c, _sc, _neg_offset);
#endif //ROUND

        _a = _a / div_factor;
        _b = _b / div_factor;
        _c = _c / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);
        _c = __CLIP_R(_c, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, _c, 0);
    }
}


/**
 * @brief Convert a vector of 32bits back to 8bit (by scaling and shifting)
 *
 * Per element k, p_res[k] = (p_in[k * stride] + bias + div_factor / 2) / div_factor
 * 
 * @warning Data must be already present in L1 memory, and the output vector must 
 * be allocated
 *
 * @param p_in Pointer to input vector a on L1 memory
 * @param len Length of the input and output vector
 * @param div_factor factor by which to divide
 * @param bias offset added before dividing by the factor
 * @param stride Collect elements from p_in with distance stride apart. Set to 1 for default stride
 * @param p_res Pointer to the output vector.
 */
void func_transform_32to8_bias(const int32_t* p_in,
                                 unsigned int len,
                                 int32_t div_factor,
                                 int32_t bias,
                                 unsigned int stride,
                                 int8_t* p_res) {

    int32_t _a, _b, _c, _d;          // temporary values
    const int32_t* _p_x = p_in;      // pointer to current element in x

#ifdef ROUND
    uint32_t _sa, _sb, _sc, _sd;                      // sign bit of a, b, c and d
    int32_t _offset = div_factor / 2;                 // offset is added every time
    int32_t _neg_offset = 1 - ((div_factor / 2) * 2); // neg offset is only added when the number if negative
#endif //ROUND

    unsigned int _num_blk = len / 4;
    unsigned int _num_rem = len % 4;

    // do the elements which can be unrolled
    while (_num_blk > 0) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);
        _c = *(_p_x + 2 * stride);
        _d = *(_p_x + 3 * stride);

        _a += bias;
        _b += bias;
        _c += bias;
        _d += bias;

#ifdef ROUND
        _sa = SIGN_BIT(_a);
        _a += _offset;
        _a = __MAC(_a, _sa, _neg_offset);

        _sb = SIGN_BIT(_b);
        _b += _offset;
        _b = __MAC(_b, _sb, _neg_offset);

        _sc = SIGN_BIT(_c);
        _c += _offset;
        _c = __MAC(_c, _sc, _neg_offset);

        _sd = SIGN_BIT(_d);
        _d += _offset;
        _d = __MAC(_d, _sd, _neg_offset);
#endif //ROUND

        _a = _a / div_factor;
        _b = _b / div_factor;
        _c = _c / div_factor;
        _d = _d / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);
        _c = __CLIP_R(_c, 127);
        _d = __CLIP_R(_d, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, _c, _d);
        
        p_res += 4;
        _p_x += 4 * stride;
        _num_blk--;
    }

    if (_num_rem == 1) {

        _a = *_p_x;
        _a += bias;
#ifdef ROUND
        _sa = SIGN_BIT(_a);
        _a += _offset;
        _a = __MAC(_a, _sa, _neg_offset);
#endif //ROUND

        _a = _a / div_factor;

        _a = __CLIP_R(_a, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, 0, 0, 0);

    } else if (_num_rem == 2) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);

        _a += bias;
        _b += bias;

#ifdef ROUND
        _sa = SIGN_BIT(_a);
        _a += _offset;
        _a = __MAC(_a, _sa, _neg_offset);

        _sb = SIGN_BIT(_b);
        _b += _offset;
        _b = __MAC(_b, _sb, _neg_offset);
#endif //ROUND

        _a = _a / div_factor;
        _b = _b / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, 0, 0);

    } else if (_num_rem == 3) {

        _a = *_p_x;
        _b = *(_p_x + 1 * stride);
        _c = *(_p_x + 2 * stride);

        _a += bias;
        _b += bias;
        _c += bias;

#ifdef ROUND
        _sa = SIGN_BIT(_a);
        _a += _offset;
        _a = __MAC(_a, _sa, _neg_offset);

        _sb = SIGN_BIT(_b);
        _b += _offset;
        _b = __MAC(_b, _sb, _neg_offset);

        _sc = SIGN_BIT(_c);
        _c += _offset;
        _c = __MAC(_c, _sc, _neg_offset);
#endif //ROUND

        _a = _a / div_factor;
        _b = _b / div_factor;
        _c = _c / div_factor;

        _a = __CLIP_R(_a, 127);
        _b = __CLIP_R(_b, 127);
        _c = __CLIP_R(_c, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(_a, _b, _c, 0);
    }

}
