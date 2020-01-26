#ifndef __NET_NET_H__
#define __NET_NET_H__

#include "rt/rt_api.h"

/*
 * Network Dimensions
 */
#define NET_F1 8
#define NET_F2 16
#define NET_D 2
#define NET_C 22
#define NET_C_ALIGN 24
#define NET_T 1125
#define NET_T_ALIGN 1128
#define NET_T8 140
#define NET_T8_ALIGN 140
#define NET_T64 17
#define NET_T64_ALIGN 20
#define NET_N 4

/*
 * Layer 1
 * =======
 * Convolution + BN
 * 
 * Input:  [C, T]
 * Weight: [F1, 64]
 * Output: [F1, C, T]
 */

#define NET_L1_PAD_START 31

#define NET_L1_PAD_END 32

#define NET_L1_PAD_INPUT_LEN 1188

#define NET_L1_PAD_INPUT_LEN_ALIGN 1188

RT_LOCAL_DATA int32_t net_l1_factor[] = { -3271, 382, 991, 312, 2647, 1090, 1291, 828 };

RT_LOCAL_DATA int32_t net_l1_offset[] = { -946, 129, -834, -281, 1437, 284, -3236, -1797 };

#define NET_L1_WEIGHT_LEN 64

RT_LOCAL_DATA int8_t net_l1_weight[] = {
    -56, 17, 1, 12, 38, -18, 23, 34, 46, 42, 54, 6, 28, 50, 4, 13, 1, -7, -25, -26, -32, -9, -17,
    -49, -24, -30, -40, 6, 33, -16, 34, 50, 36, 14, 68, 61, 80, 14, 38, 52, -14, 17, 6, -9, -34,
    -38, -28, -38, -32, -37, -48, -37, -38, 12, -1, -38, 24, 4, 53, 14, 60, 59, 47, 54, -38, -17,
    -29, -36, 11, 32, 37, 2, 56, 48, 29, 6, 33, 9, -13, -32, -26, -47, -30, -41, -26, -13, -12, 7,
    7, -6, 39, 22, 43, 41, 51, 16, 29, 7, 3, -42, -24, -64, -26, -48, 6, -34, 15, -32, 34, -11, 16,
    40, 53, 63, 42, 20, 4, 2, -31, -9, -47, -43, -37, -55, -21, 3, 32, 58, 64, 66, 19, 12, 31, 27,
    42, 40, 30, -31, 28, 7, 43, 55, 22, 64, 35, 35, 67, 31, 35, 36, 19, -20, 23, 44, 70, 50, 38,
    14, 8, 27, 53, 22, 15, 38, -8, 44, 30, -17, 6, 35, 48, -7, 18, 13, -19, 55, 11, 45, -13, -8,
    -38, -35, -5, 31, -3, -13, -2, 25, 19, 8, -39, -10, -43, -69, -22, -7, -50, 8, 16, 5, 62, 31,
    37, 52, 31, -7, -30, -9, -55, -16, -51, -41, -68, -8, -23, -13, 32, 20, 31, 36, 18, 37, 25, 24,
    2, 21, -30, -39, 1, -50, -58, -12, -30, -45, -54, 6, 42, 14, 52, 64, 41, 43, 54, -8, 9, 6, -44,
    -30, -53, -57, -33, 12, -14, 19, 36, 64, -87, -71, -65, 7, -25, -46, -4, -32, -21, -32, -20,
    -43, -39, -35, -37, 3, 17, -43, -14, -21, 5, -23, -16, -18, -5, 4, -46, -10, -27, -31, -6, -55,
    10, 0, -45, -23, -16, -8, -46, -4, -41, -8, 24, -44, 28, -11, -2, -27, -42, -45, 6, -36, -8,
    -36, -22, 20, -20, -5, -48, -36, 2, -48, -62, -35, 127, 34, 37, 30, 26, 42, 30, 28, 58, 24, 50,
    64, 19, 30, 20, 5, 43, 51, 15, 53, -7, 46, 34, -19, 34, 20, 41, 48, 37, -6, -3, 27, 44, 52, 25,
    24, 14, 32, 49, 42, -9, 3, 45, -11, 44, -23, 40, 18, 19, 11, 17, 3, 60, -18, 19, 45, 9, 42, 27,
    -7, 11, 46, -15, 12, 18, -6, 36, 40, -19, -14, 17, 18, 26, 10, 6, 41, 7, 43, 25, 9, 37, 42, 29,
    18, 14, 10, 23, 29, 7, 42, -8, 60, 18, 49, 24, 17, 46, 45, 26, 15, 52, 63, -11, 43, 17, 2, 20,
    22, 36, 31, 48, -11, 10, 39, 17, 46, 48, 52, 9, 6, 11, -13, 10, 23, 45, 27, 65, 48, 41, 50, 41,
    37, 5, -4, 43, 7, 31, 10, 0, 19, 57, 58, -3, 35, 0, 23, 24, 11, 36, -23, 15, 3, 31, 13, 42, 3,
    58, 1, 14, 42, 30, 3, 41, 4, 63, 3, 9, 19, 36, 9, 42, -1, 38, 18, 50, 13, 14, 45, 23, 48, 10,
    20, 60, 8, 40, 35, 43, -3, 34, 20, 64, 60
};

/*
 * Layer 2
 * =======
 * Convolution + BN + ReLU + Pooling
 * 
 * Input:  [F1, C, T]
 * Weight: [F2, C] (aligned to [F2, 24]
 * Output: [F2, T // 8]
 */

RT_LOCAL_DATA int32_t net_l2_factor[] = {
    34, 7, 35, 20, 34, 22, 90, 236, 20, 12, 32, 51, 19, 37, 42, 84
};

RT_LOCAL_DATA int32_t net_l2_offset[] = {
    -364, -163, -532, -362, -453, -131, -467, -1855, -51, -111, -319, -392, -187, -317, -108, 42
};

#define NET_L2_WEIGHT_LEN 24

RT_LOCAL_DATA int8_t net_l2_weight[] = {
    8, 42, -11, 77, -7, 30, -54, -25, 80, 21, 60, 7, -64, -84, 40, 55, 53, -11, -92, -52, -30, -99,
    0, 0, -26, -25, -12, 68, -88, -57, -53, 71, 30, -4, 74, 69, 31, -59, 58, 46, 17, -21, -60, -3,
    -34, 4, 0, 0, 1, 60, -56, 40, -101, -7, 15, 25, -62, 70, -21, 34, 120, -44, 11, 10, -72, 37,
    31, -88, 32, -15, 0, 0, 7, -10, 21, -61, -39, 28, 11, 16, 82, 10, 17, 8, -108, -5, 61, -27,
    -21, 49, 25, -33, -24, -4, 0, 0, -14, -5, 54, 0, 70, -50, 26, 85, -37, -3, -82, -13, 64, 17,
    -27, -97, 29, -40, 4, 69, -69, 26, 0, 0, 72, 22, -36, -40, 11, 2, -11, -45, -25, -29, -24, 23,
    32, -76, -22, 62, 28, -15, 17, -46, 87, 17, 0, 0, -3, -13, -60, 31, 127, 10, -6, 25, 1, -60,
    105, -78, -27, -18, -22, -1, -70, 23, -25, 15, 17, 34, 0, 0, 61, -12, -89, 28, -112, -97, 5,
    29, 30, 12, 37, -9, 50, -23, -16, -3, 89, 15, 34, 6, -22, -89, 0, 0, 75, 115, -62, -61, 21,
    -110, 58, -8, -100, 1, 25, -79, 34, -29, 22, 8, -26, 24, 5, 55, -5, -26, 0, 0, 56, -13, -104,
    72, -9, -36, -45, 24, 3, 4, -33, -34, 3, 15, 48, 74, -17, -52, -23, 50, 26, 33, 0, 0, -33, -6,
    48, -58, 46, -9, 120, 20, -60, -17, -15, 17, 45, -61, -87, -30, -4, -38, 8, -3, 72, 18, 0, 0,
    -63, -73, 16, -38, 19, 5, -77, 61, 29, -33, 16, 24, 50, 57, 60, -29, -30, -59, -10, 38, -22, 0,
    0, 0, 22, 22, 64, 25, -73, 16, 31, -62, -23, -35, -84, 7, 58, 24, -35, -30, 13, -29, -44, 52,
    63, 9, 0, 0, 60, -35, 2, -25, -25, -68, -29, 56, 33, 5, 14, -10, 31, -23, 43, 2, 113, 1, -52,
    32, -34, -105, 0, 0, 5, -2, 56, 59, -3, 71, -124, -93, -9, -33, -10, 48, 51, 11, -19, 27, 25,
    13, -32, -22, -23, 32, 0, 0, 62, -30, 9, -34, 58, -8, 23, -18, -32, 8, -70, -36, 78, 42, -87,
    5, 17, -35, -53, -37, 93, 78, 0, 0
};

/*
 * Layer 3
 * =======
 * Convolution
 * 
 * Input:  [F2, T // 8]
 * Weight: [F2, 16]
 * Output: [F2, T // 8]
 */
int32_t net_l3_factor = 391;

#define NET_L3_WEIGHT_LEN 16

RT_LOCAL_DATA int8_t net_l3_weight[] = {
    -107, -64, -93, -59, -62, -40, -50, -62, -69, -30, -33, -1, -21, 26, 47, 45, -14, -5, -32, -18,
    -11, -23, -30, -31, -44, -42, -52, -63, -55, -60, -76, -77, 81, 68, 46, 31, 41, 37, 30, -4, 24,
    38, 17, 25, 34, 49, 50, 65, -48, -49, -42, -31, -22, -31, -33, -32, -28, -29, -29, -33, -31,
    -38, -37, -52, -43, -48, -50, -51, -45, -33, -37, 14, -22, -16, 13, -26, -8, 6, -2, -11, 11,
    -19, -66, -55, -46, -37, -54, -35, -13, -19, -11, -8, 1, -7, 23, 40, -104, -82, -48, -47, -46,
    -20, -26, -29, -35, -6, -18, -31, -50, -43, -65, -83, -64, -32, -11, -8, -25, -26, -50, -56,
    -52, -43, -50, -71, -106, -115, -67, -91, 33, 36, 42, 39, 30, 41, 47, 69, 71, 39, 29, 17, 12,
    -17, -21, -8, -54, -47, -25, -27, -1, -5, -20, -9, -26, -26, -23, -27, -1, -35, 10, 14, 127,
    65, 34, 33, 39, 38, -4, -5, 22, -18, 17, -41, -34, 23, -1, 56, -25, -32, -17, -46, -43, -58,
    -54, -33, -19, -27, -9, -26, -6, -30, -23, -59, -54, -28, -11, -13, 28, 19, 22, 32, 22, 24, 25,
    11, 24, 29, 25, 29, -17, 8, -28, 7, 3, -5, -28, -45, -30, -26, -57, -79, -76, -61, -33, -61,
    -26, -19, -22, -15, -7, 9, -26, -14, -37, -24, -30, -49, -56, -44, -65, -50, 23, 51, 47, 25,
    29, 29, 37, 32, 27, 33, 34, 33, 34, 35, 65, 67
};

/*
 * Layer 4
 * =======
 * Convolution + BN + ReLU + Pooling
 * 
 * Input:  [F2, T // 8]
 * Weight: [F2, F2]
 * Output: [F2, T // 64]
 */

RT_LOCAL_DATA int32_t net_l4_factor[] = {
    62, 40, 41, 64, 57, 55, 64, 47, 60, 58, 41, 48, 80, 49, 64, 60
};

RT_LOCAL_DATA int32_t net_l4_offset[] = {
    465, -1225, 212, 332, 1547, -103, -22, 655, -1478, 58, 597, -72, 1034, 666, 1825, 832
};

#define NET_L4_WEIGHT_LEN 16

RT_LOCAL_DATA int8_t net_l4_weight[] = {
    31, -33, -10, -61, 2, 56, 39, 49, -85, 15, -54, -51, 34, -23, -24, -8, -67, 18, 10, -48, -19,
    -36, -16, -94, 26, -19, 4, -27, -6, -18, -39, -8, 1, -17, 41, 52, -19, 4, -57, -17, 9, -2, -21,
    -59, 1, 31, 42, 14, -12, 62, -71, 76, -53, -47, 47, -77, 18, -12, 57, -38, 32, -18, 14, 13, 51,
    58, 5, 26, 36, 6, 29, -61, 10, 17, -19, 37, -5, -57, -8, -127, -103, 69, 41, 8, -24, 50, 16,
    45, 61, 16, 50, 22, -5, -65, -31, 8, 32, 49, 8, -21, 25, -20, 56, 75, 39, -19, -3, 83, 51, 9,
    -65, 78, 20, 24, -52, -55, 37, -44, 55, 106, -58, -56, -59, 8, 2, -15, 54, 18, -5, -92, 55,
    -45, 13, -47, -66, -27, 40, -29, 85, 28, -17, -15, 20, 21, 75, 14, -28, -10, -74, 12, -28, 30,
    8, -54, 67, -23, -30, 40, 12, -49, -12, 12, -34, -57, -10, 29, 72, -23, 1, -22, 16, 7, 26, 36,
    37, -64, 16, -19, 122, -11, 47, 1, -64, -76, -12, -3, -28, 3, 6, 3, 0, -54, 76, 10, 21, 114,
    20, -12, -33, -44, -31, -18, 35, 4, 8, -9, -53, -49, 13, -25, -7, 28, 25, -39, -26, -21, -93,
    -23, 27, -19, 22, 105, -32, -20, -31, 92, -24, 83, -5, 76, 59, 21, 31, -8, -38, -12, 3, 23, -9,
    -20, -25, 3, -27, -35, -17, -17, 34, 78, -10, 102, 19, -19, -94, -1, -25, 2
};

/*
 * Layer 5
 * =======
 * Linear Layer (without scaling in the end)
 * 
 * Input:  [F2, T // 64]
 * Weight: [N, F2 * (T // 64)]
 * Bias:   [N]
 * Output: [N]
 */

RT_LOCAL_DATA int8_t net_l5_bias[] = { -14, 7, -11, -14 };

#define NET_L5_WEIGHT_LEN 272

RT_LOCAL_DATA int8_t net_l5_weight[] = {
    45, -3, 31, 18, 8, -12, -22, 9, 21, 27, 18, -31, -3, 5, 7, 19, -6, 12, 22, 5, -16, -30, -4,
    -12, -17, 5, -28, -24, 0, -22, 0, -23, -27, 2, 7, 22, 31, 36, 10, -36, -24, -48, -84, -21, 2,
    -32, -39, -28, -36, -41, -50, -37, -4, 19, -7, 9, -3, -7, -13, 9, 23, 18, 5, 4, 42, 24, 1, 36,
    14, -2, -36, -16, 16, 22, -6, -4, -13, -27, -7, 25, 10, 4, 4, 13, 4, -27, 11, 4, -26, 9, 20,
    -1, -10, 3, 1, -17, -33, 1, 0, -12, 13, 2, -11, -13, -17, -10, 42, 36, 37, 39, 10, 7, 16, 22,
    17, -3, -14, -6, -3, -16, -2, 11, 93, 32, 22, 14, 30, 6, 19, 13, 26, 22, 33, 11, 6, -5, -5,
    -22, -24, -29, -11, -13, -59, -71, -41, -17, -28, -1, 5, -22, -15, -4, 1, 17, -12, -12, -60,
    32, 6, 11, -8, -12, -30, -6, 4, -10, -40, -16, -10, -4, 0, -5, -29, -20, 6, 24, 31, 30, 15, 10,
    11, 11, 20, 18, 3, 18, 20, -23, -2, -15, -33, -65, -115, -90, -55, -59, -37, -30, -43, -32,
    -25, -1, -22, -38, 13, -3, -30, -55, -33, -9, 9, 3, 14, -7, -18, -9, 0, -12, -13, -27, 0, 5,
    -2, -2, 16, -50, -12, 22, -3, 16, 2, -23, -57, -45, -13, -10, -20, -19, 15, 7, -12, -5, -32, 4,
    7, 28, 12, 14, 26, 24, 20, 4, 15, 26, 3, -8, 9, 68, -27, 12, 11, 10, 27, 70, 14, 16, 41, 11, 9,
    47, -6, 9, -11, 2, -5, -24, -30, -35, 6, -26, -74, -63, -46, -86, -102, -61, -67, -48, -29, 14,
    -9, 19, -10, -15, -98, -89, -56, -17, -43, -35, -34, -34, -10, -8, -30, -33, -14, -7, -19, -49,
    -1, 62, 30, 30, 32, 37, 15, 18, 14, 27, 51, 32, 15, 12, -20, -43, 0, 3, 25, 47, 51, 22, 29, 32,
    34, 33, -7, 18, 6, -12, -9, -18, 6, 17, 15, 22, 48, 44, 32, 21, 36, 7, 1, -3, -14, -13, 17, 15,
    29, -9, -26, 35, 5, -16, -10, -19, -39, -13, -2, -12, 28, 33, -23, -17, -9, 2, 51, 0, -39, 3,
    -6, -4, -4, -5, -37, -30, -33, 1, 26, 17, 0, 18, -12, -6, -55, -86, -85, -29, -50, -22, -14,
    -45, -34, -36, -49, -51, -43, -18, -20, 21, 31, 46, 0, -69, -81, -28, -21, -61, -2, -25, -26,
    -2, 1, -21, -2, -21, 0, -58, 8, 5, 13, -18, -24, -5, -9, -11, 14, 20, 8, 16, 12, -4, 4, 1, -7,
    -29, -61, -87, -68, -80, -56, -52, -55, -32, -6, -38, -57, -30, -23, 33, -12, 21, 26, 13, -41,
    -39, -23, -15, -4, -38, -7, -5, -21, -26, -25, -2, -14, 1, 38, 63, 25, 5, 21, 35, 35, 33, 38,
    42, 34, 20, 5, 25, 32, -15, -27, 33, 27, 0, 13, -1, 20, 18, 17, 29, 50, 38, 21, 0, 30, 26, -11,
    10, -10, -44, 64, 45, 22, 3, 15, 17, 5, 5, -2, 11, 14, 10, 11, 26, 13, -63, 34, 17, -17, -11,
    23, -41, -20, -3, -20, 8, 5, -12, -14, -30, -29, -31, -37, -24, 2, -4, -3, -26, -9, 1, -2, 25,
    41, 8, 3, -13, -27, -12, -18, 2, 19, 30, 8, 2, -1, 16, 5, 6, 2, 0, 12, 15, 21, 25, 26, -9, 4,
    -4, -42, -10, 12, 14, 15, -8, -14, -14, -1, 4, 8, 17, 28, 7, 18, -2, -12, 1, -5, -37, -28, -26,
    -47, -29, -21, -18, -32, -21, -34, -24, -4, 12, 39, 31, 5, -53, -127, -89, -58, -7, -32, -8,
    -10, 2, 5, 22, -20, -21, -28, -45, -1, -35, 2, 11, 34, 10, 19, 32, 13, 8, -32, -15, 26, 4, -11,
    -16, -10, -1, -36, -13, 17, 20, 5, -11, 20, -8, -15, -7, -24, -44, -24, -17, -23, -14, -30, -7,
    30, 14, 0, 28, -7, 23, 0, -12, 7, 16, -3, 15, 18, 11, 17, -5, 19, 20, 18, 19, 22, 18, -7, 13,
    21, 3, 10, 8, 16, -1, 13, 29, -16, -39, 22, 35, 48, 16, 50, 28, 1, 3, 14, -6, -19, -21, 23, 35,
    46, 18, -10, -29, -12, 13, -6, -20, -1, -6, 1, 6, 8, -12, -16, 9, -13, -11, 27, -6, 2, -1, -10,
    5, 6, 13, 16, -3, -6, -7, -11, 7, -28, 12, 21, 4, 45, 5, -33, -31, -32, -14, -45, -11, -9, 15,
    44, -3, -14, 8, 39, -21, -34, 32, 6, -11, 15, 28, -15, 7, 23, 0, -2, 2, 23, 24, 10, -12, 24,
    42, -53, -87, -28, -14, 10, -24, -17, -36, 0, 6, -22, -39, -11, -30, -42, -16, -13, 9, 24, -3,
    -83, -69, 28, -5, -12, -9, 5, -12, -2, 16, 15, 11, 4, 8, 1, 7, 3, 31, 26, -12, 15, 15, 13, 29,
    12, 10, 47, 2, 2, -2, 6, 21, 39, 38, 16, 7, 15, 18, 8, 20, 7, 16, 5, -8, -17, -14, -5, 25, 29,
    -6, -26, -14, 3, -23, 5, 27, 22, 3, -33, -6, 7, 6, 4, -4, -1, -28, -33, -30, -31, -90, -66,
    -75, -37, -68, -52, -37, 16, 1, 9, 0, -20, -40, -71, -37, 33, 33, 34, -4, -14, -78, -7, 11, 6,
    -14, -17, 28, 7, 45, 43, -22, -30, 9, -5, -16, 14, -3, 15, 7, 15, -5, -16, 21, 14, 15, -18, 2,
    24, -36, -102, -75, -26, 20, -18, 0, 15, -20, 9, 3, 24, 11, 18, 16, -23, 23, 0, 3, -10, -22,
    -10, -10, 5, 15, -2, 0, 29, -11, 0, -9, -15, 12, -11, 25, 26, 13, 10, 21, 17, 12, -1, 6, 15,
    -2, -15, 1, -3, 19, 39, -4, -34, -53, -22, 0, -16, -12, -26, -10, 9, 16, 20, 25, -36, -23, -14,
    1, 18, 41, 9, -13, 26, 35, 2, 13, -6, 0, 6, 1, 7, -9, 9, 2, -12, 22, 37, 25, 35, 32, 13, 4, 21,
    20, 24, 32, 17, 18, 27, -1, -41, -41, -27, -20, -2, -7, -50, -22, -5, -15, -29, -19, -28, -29,
    -19, -12, -23, 4, 9, 22, -12, -17, -31, -75, -13, -7, 0, -7, -6, 6, -30, -22, -26, 1, -4, -7,
    16, 10, -51, -57, -37, -39, -42, -29, -52, -56, -19, -28, -30, 18, -1, 22, -9, -71, -42, -14,
    33, 18, -44, 6, -22, -20, -19, -48, -3, -12
};

#endif//__NET_NET_H__