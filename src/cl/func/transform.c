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

    int32_t a, b, c, d;             // temporary values
    const int32_t* p_x = p_in;      // pointer to current element in x
    int8_t* p_y = p_res;            // pointer to the output element

#ifdef ROUND
    uint32_t sa, sb, sc, sd;                          // sign bit of a, b, c and d
    int32_t offset = div_factor / 2;                  // offset is added every time
    int32_t neg_offset = -((div_factor / 2) * 2) + 1; // neg offset is only added when the number if negative
#endif //ROUND

    unsigned int num_blk = len / 4;
    unsigned int num_rem = len % 4;

    // do the elements which can be unrolled
    while (num_blk > 0) {

        a = *p_x;
        b = *(p_x + 1 * stride);
        c = *(p_x + 2 * stride);
        d = *(p_x + 3 * stride);

        p_x += 4 * stride;

#ifdef ROUND
        sa = SIGN_BIT(a);
        a += offset;
        a = __MAC(a, sa, neg_offset);

        sb = SIGN_BIT(b);
        b += offset;
        b = __MAC(b, sb, neg_offset);

        sc = SIGN_BIT(c);
        c += offset;
        c = __MAC(c, sc, neg_offset);

        sd = SIGN_BIT(d);
        d += offset;
        d = __MAC(d, sd, neg_offset);
#endif //ROUND

        a = a / div_factor;
        b = b / div_factor;
        c = c / div_factor;
        d = d / div_factor;

        a = __CLIP_R(a, 127);
        b = __CLIP_R(b, 127);
        c = __CLIP_R(c, 127);
        d = __CLIP_R(d, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(a, b, c, d);
        
        p_res += 4;
        num_blk--;
    }

    if (num_rem == 1) {

        a = *p_x;

#ifdef ROUND
        sa = SIGN_BIT(a);
        a += offset;
        a = __MAC(a, sa, neg_offset);
#endif //ROUND

        a = a / div_factor;

        a = __CLIP_R(a, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(a, 0, 0, 0);

    } else if (num_rem == 2) {

        a = *p_x;
        b = *(p_x + 1 * stride);

#ifdef ROUND
        sa = SIGN_BIT(a);
        a += offset;
        a = __MAC(a, sa, neg_offset);

        sb = SIGN_BIT(b);
        b += offset;
        b = __MAC(b, sb, neg_offset);
#endif //ROUND

        a = a / div_factor;
        b = b / div_factor;

        a = __CLIP_R(a, 127);
        b = __CLIP_R(b, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(a, b, 0, 0);

    } else if (num_rem == 3) {

        a = *p_x;
        b = *(p_x + 1 * stride);
        c = *(p_x + 2 * stride);

#ifdef ROUND
        sa = SIGN_BIT(a);
        a += offset;
        a = __MAC(a, sa, neg_offset);

        sb = SIGN_BIT(b);
        b += offset;
        b = __MAC(b, sb, neg_offset);

        sc = SIGN_BIT(c);
        c += offset;
        c = __MAC(c, sc, neg_offset);
#endif //ROUND

        a = a / div_factor;
        b = b / div_factor;
        c = c / div_factor;

        a = __CLIP_R(a, 127);
        b = __CLIP_R(b, 127);
        c = __CLIP_R(c, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(a, b, c, 0);
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

    int32_t a, b, c, d;             // temporary values
    const int32_t* p_x = p_in;      // pointer to current element in x
    int8_t* p_y = p_res;            // pointer to the output element

#ifdef ROUND
    uint32_t sa, sb, sc, sd;                          // sign bit of a, b, c and d
    int32_t offset = div_factor / 2;                  // offset is added every time
    int32_t neg_offset = -((div_factor / 2) * 2) + 1; // neg offset is only added when the number if negative
#endif //ROUND

    unsigned int num_blk = len / 4;
    unsigned int num_rem = len % 4;

    // do the elements which can be unrolled
    while (num_blk > 0) {

        a = *p_x;
        b = *(p_x + 1 * stride);
        c = *(p_x + 2 * stride);
        d = *(p_x + 3 * stride);

        a += bias;
        b += bias;
        c += bias;
        d += bias;

#ifdef ROUND
        sa = SIGN_BIT(a);
        a += offset;
        a = __MAC(a, sa, neg_offset);

        sb = SIGN_BIT(b);
        b += offset;
        b = __MAC(b, sb, neg_offset);

        sc = SIGN_BIT(c);
        c += offset;
        c = __MAC(c, sc, neg_offset);

        sd = SIGN_BIT(d);
        d += offset;
        d = __MAC(d, sd, neg_offset);
#endif //ROUND

        a = a / div_factor;
        b = b / div_factor;
        c = c / div_factor;
        d = d / div_factor;

        a = __CLIP_R(a, 127);
        b = __CLIP_R(b, 127);
        c = __CLIP_R(c, 127);
        d = __CLIP_R(d, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(a, b, c, d);
        
        p_res += 4;
        p_x += 4 * stride;
        num_blk--;
    }

    if (num_rem == 1) {

        a = *p_x;
        a += bias;
#ifdef ROUND
        sa = SIGN_BIT(a);
        a += offset;
        a = __MAC(a, sa, neg_offset);
#endif //ROUND

        a = a / div_factor;

        a = __CLIP_R(a, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(a, 0, 0, 0);

    } else if (num_rem == 2) {

        a = *p_x;
        b = *(p_x + 1 * stride);

        a += bias;
        b += bias;

#ifdef ROUND
        sa = SIGN_BIT(a);
        a += offset;
        a = __MAC(a, sa, neg_offset);

        sb = SIGN_BIT(b);
        b += offset;
        b = __MAC(b, sb, neg_offset);
#endif //ROUND

        a = a / div_factor;
        b = b / div_factor;

        a = __CLIP_R(a, 127);
        b = __CLIP_R(b, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(a, b, 0, 0);

    } else if (num_rem == 3) {

        a = *p_x;
        b = *(p_x + 1 * stride);
        c = *(p_x + 2 * stride);

        a += bias;
        b += bias;
        c += bias;

#ifdef ROUND
        sa = SIGN_BIT(a);
        a += offset;
        a = __MAC(a, sa, neg_offset);

        sb = SIGN_BIT(b);
        b += offset;
        b = __MAC(b, sb, neg_offset);

        sc = SIGN_BIT(c);
        c += offset;
        c = __MAC(c, sc, neg_offset);
#endif //ROUND

        a = a / div_factor;
        b = b / div_factor;
        c = c / div_factor;

        a = __CLIP_R(a, 127);
        b = __CLIP_R(b, 127);
        c = __CLIP_R(c, 127);

        *((int32_t*)p_res) = (int32_t)__PACK4(a, b, c, 0);
    }

}
