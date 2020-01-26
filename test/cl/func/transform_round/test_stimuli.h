#ifndef __TEST_STIMULI_H__
#define __TEST_STIMULI_H__

#include "rt/rt_api.h"

#define LENGTH 1027

int32_t div_factor = 10;

int32_t bias = 50;

RT_LOCAL_DATA int32_t vec_x[] = {
    281, -1193, 359, 80, -2364, -1336, -431, 984, 1567, 461, 1496, 108, -2078, 156, -2011, 479,
    1171, 2345, -339, -1001, 1974, -1224, -2481, -60, 2449, -2362, 1811, -230, -181, 1826, 427,
    1710, -1571, 1096, 822, 1084, -1164, -2190, 1195, -92, 1905, 977, 2409, 1143, -2531, 641, -789,
    2383, 1369, 895, 2019, -2002, 628, 2031, -1372, -1354, 1279, -2203, -1711, 596, -1889, 125,
    -2303, 1774, 2065, -2046, -1010, 1157, 612, 2277, 2452, 1149, 725, -1112, 1998, 341, -1697,
    -697, 2511, 1233, -25, -1181, 1676, 1240, 1795, 600, 2184, 2374, 1038, 267, 1835, -1730, -2446,
    1827, 309, 881, -356, -30, -1605, -2537, -2392, -52, 1227, 362, -931, 1094, -2436, -636, 1908,
    744, -2064, 904, -1463, 605, 959, -2100, 2235, -1869, -1442, 1502, -909, -454, 1341, -522,
    2498, 2268, 1916, 152, -1745, -1703, -1258, -1675, 504, 521, 1000, 2043, -1660, 579, 666,
    -2022, -1057, -1353, 1522, 2251, -2250, 1956, -1388, 2093, 204, -1494, 1740, -276, -1580,
    -1738, 334, -2495, 1946, -1295, 876, -620, 2238, -603, 1585, -1683, 1687, -765, -107, 2048,
    1670, 1447, 1146, -1286, -672, 467, 2106, 1743, 72, -2392, -2077, 162, -1803, 1915, -532,
    -2202, 178, 1311, 324, -910, 129, -2278, 878, -2008, -2388, -2323, 1147, -1986, -2357, -1352,
    800, 389, 1965, -1408, -1268, -2385, 1078, 673, -2466, 2305, -1537, -964, -1595, -572, 1686,
    244, -1938, -1454, 1802, -1590, -1989, 1321, 1244, -1402, 724, 2438, -2138, -923, 1614, 2034,
    -720, 1658, -856, 2110, -498, 1620, -738, -979, -2313, 2455, 235, -2422, 1489, -2139, 1819,
    -889, 2333, 943, 925, -1819, 2077, 1272, 1105, 853, 1928, 2543, -230, 2131, 761, -2147, 1004,
    -1408, -782, 461, 877, 1047, 1615, 1763, -640, 2114, 399, 593, -2184, 2438, -723, -968, -1784,
    -2512, -845, 2157, 1525, 282, -1908, -1687, 1881, -673, -1582, 22, -646, -2389, -1359, 2170,
    -2095, -1258, 2368, -407, -2518, 1464, 577, 1264, 746, -1638, 2180, -2112, -630, -954, -1164,
    2430, 146, -996, -1200, 13, 1187, 998, -138, 804, -1932, -1019, 1193, -248, -1194, 259, 625,
    543, 2388, -2080, 53, -2164, 2372, -363, -2058, -2371, 1448, 923, 2228, -369, 1489, 355, -628,
    1525, 1675, -2320, 1146, 673, -2366, 2486, 2548, 817, 838, -786, 1417, -1693, 1528, -1147, 240,
    1543, 499, 889, -319, -2487, 2252, -156, 1415, -147, 1906, 675, -219, 1963, 356, 1001, 2427,
    -532, -439, -539, -324, -52, -865, 2202, 1631, -31, 1300, -2263, 1396, -1263, 410, -963, 1524,
    818, -2312, 1434, -906, 166, -320, -1827, 1374, 611, 651, 1114, 1748, 2553, -318, 2025, -154,
    -298, 1941, -1273, 1711, 2370, -126, 331, 2154, 740, 1450, 1162, -2217, 748, -1433, -1001,
    2510, 382, -1716, -493, -1515, -1914, 1831, -2138, 1840, 1941, -65, -430, 2058, 2459, 2199,
    2266, -1080, 1153, 1480, -823, 1303, -1926, 1211, -214, 1842, -1894, -2318, 1755, -2102, 1398,
    1467, 1439, 2533, 1093, -1313, -220, -974, 1714, -1804, -1580, 2214, 382, 923, 243, 758, -1044,
    -2205, 2526, 392, -2157, -1518, -112, -2060, 1090, 859, -1878, -644, 1811, -2139, 2328, -1456,
    -1615, -910, 616, 1632, 112, -2333, -1729, 873, 376, 1165, 1379, -1909, 1320, 1624, 1614, 2238,
    395, 1040, 2122, 1130, 2236, 1750, -2415, 529, -2492, -263, -2343, -1836, 1611, -1978, -720,
    -2174, 1049, 2054, -158, -1237, 1612, 46, -541, 243, 1015, -2129, 841, 737, 514, 2347, -1310,
    2211, 1089, 719, -1626, -440, -1022, -1361, 1778, -1814, 1741, 1783, -250, 1720, -1357, -484,
    771, 597, 575, 380, 1657, -2226, 913, -65, 1251, -2027, 2215, 2200, -1758, -583, -221, -1269,
    -448, 1674, -64, 1618, -2219, 426, 2036, 1665, -2413, -2094, 810, 1105, 192, 2534, -2481, 1227,
    599, -1202, 1482, -1472, -2368, -1498, 1874, -1523, -1734, 254, -2218, -678, -519, -2294, 1956,
    2071, -710, 1220, 1188, -325, -1715, -2006, -1892, -2086, -295, -2552, -266, 1796, 1326, 248,
    -1608, 1414, 728, 1776, 1726, 805, -32, -2268, 1252, -282, 1182, -443, 1991, -850, -976, -454,
    2302, 1400, -1905, -670, 1419, -834, -94, 1381, -1765, -1058, -212, 1086, -2126, -1120, 13,
    -153, -762, -1836, -367, -792, -1277, 1274, -2466, -2067, -67, 1924, -1713, -1459, -1735, 1978,
    1453, -2431, -1181, 786, 2228, -2369, -1754, -1926, -160, 558, 1794, 2367, -2479, 1183, -2209,
    -1151, -1952, 2126, 997, -1717, -2488, 923, -2411, -2283, 1111, -2148, 456, 1651, -2560, 832,
    137, -1793, -1128, 101, 1134, 1297, 1505, -728, 2127, -2443, 1272, -543, 932, 2403, -274,
    -1935, 1893, -2481, 1515, -1891, 236, -1722, 932, 1344, 987, 2374, -1599, -2259, 0, 287, 2262,
    -642, 702, 1757, -1123, 916, 1563, -1165, 1666, -81, -833, 2277, -1958, 275, 1411, 1303, -535,
    -383, 1345, 2030, -767, 821, 1436, -660, -1209, 2360, 1346, 1396, 1428, 1519, -1031, -1042,
    747, 2411, -1080, -2176, 2408, 387, -911, -500, 626, 355, -844, -600, -745, 2315, -1391, -2207,
    2278, 2210, 1796, 1690, 1788, 1289, 1654, -291, -2518, -2118, -1846, -2244, -1423, -1232,
    -2054, -329, -1927, -325, 2314, 1876, -1340, -1540, 286, -1749, 275, -723, 2403, 524, -113,
    -1518, -2538, 2350, -1031, 2428, 64, 1857, 1155, -2148, 446, 910, 721, -1272, -1563, -1318,
    -2404, -1132, -2156, 1345, 771, -2448, -1682, 252, -590, 69, 518, -297, -1523, -524, -592, 266,
    2035, 1248, -1410, -91, -84, 722, -1313, 1316, 524, 961, 493, 1053, -970, 2092, -1216, 1651,
    -438, -132, -40, 1633, -866, 2080, 1524, 379, 1634, -711, -674, 1854, 341, -2499, -1316, -7,
    -906, -1894, 18, 838, 1436, -2218, -1588, -1818, 670, -1016, -1779, -783, -505, -1717, 949,
    -1565, -2423, -2175, -438, -1149, 925, 1683, -2406, -601, -872, 713, 1743, -420, 350, 1259,
    -892, -1635, -1496, -779, -388, 1558, 1335, -1092, -1148, -19, -1229, 1199, -2437, 1178, -1310,
    -1756, -1235, -200, -1050, -1893, -2341, -2418, -2259, -1818, -2077, -4, -579, 973, -2304,
    1563, 908, -1931, 1353, -294, -2225, 253, 1202, 208, 1068, 312, 2225, 541, 461, 1929, 480, 74,
    2415, -1045, 914, -538, 2515, -659, -1703, -2356, -1878, -1153, -1121, -117, 1313, -2189, 1326,
    -64, 738, -836, -404, 1575, -1230, -1259, 1937, -385, -1896, 203, -1427, 2116, -454, 2137,
    -569, -2113, 1297, 1836, -472, 1315, 1124, -344, 755, 1698, -1837, -1450, 1859, 2148, -2408,
    -707, 148, 2527, 2343, 1222, -2540, 802, -1445, 414, 1280, -1233, -246, -220, -951, -1097,
    2083, -1687, -2036, 980, 11, 38, -14, 1975, 1194, -1705, -1205, 1599, -2007, -1471, 1875, -701,
    915, -530, 92, -1228, -1368, -2071, -1723, 836, 910, 1900, 1089, 2467, -1029, -1405, 2493,
    2463, 822, 2070, 1517, 2021, -515, -2197, 938, -1698, -1319, -938, -1850, 1686
};

RT_LOCAL_DATA int8_t vec_exp[] = {
    28, -119, 36, 8, -128, -128, -43, 98, 127, 46, 127, 11, -128, 16, -128, 48, 117, 127, -34,
    -100, 127, -122, -128, -6, 127, -128, 127, -23, -18, 127, 43, 127, -128, 110, 82, 108, -116,
    -128, 120, -9, 127, 98, 127, 114, -128, 64, -79, 127, 127, 90, 127, -128, 63, 127, -128, -128,
    127, -128, -128, 60, -128, 13, -128, 127, 127, -128, -101, 116, 61, 127, 127, 115, 73, -111,
    127, 34, -128, -70, 127, 123, -2, -118, 127, 124, 127, 60, 127, 127, 104, 27, 127, -128, -128,
    127, 31, 88, -36, -3, -128, -128, -128, -5, 123, 36, -93, 109, -128, -64, 127, 74, -128, 90,
    -128, 61, 96, -128, 127, -128, -128, 127, -91, -45, 127, -52, 127, 127, 127, 15, -128, -128,
    -126, -128, 50, 52, 100, 127, -128, 58, 67, -128, -106, -128, 127, 127, -128, 127, -128, 127,
    20, -128, 127, -28, -128, -128, 33, -128, 127, -128, 88, -62, 127, -60, 127, -128, 127, -76,
    -11, 127, 127, 127, 115, -128, -67, 47, 127, 127, 7, -128, -128, 16, -128, 127, -53, -128, 18,
    127, 32, -91, 13, -128, 88, -128, -128, -128, 115, -128, -128, -128, 80, 39, 127, -128, -127,
    -128, 108, 67, -128, 127, -128, -96, -128, -57, 127, 24, -128, -128, 127, -128, -128, 127, 124,
    -128, 72, 127, -128, -92, 127, 127, -72, 127, -86, 127, -50, 127, -74, -98, -128, 127, 24,
    -128, 127, -128, 127, -89, 127, 94, 93, -128, 127, 127, 111, 85, 127, 127, -23, 127, 76, -128,
    100, -128, -78, 46, 88, 105, 127, 127, -64, 127, 40, 59, -128, 127, -72, -97, -128, -128, -84,
    127, 127, 28, -128, -128, 127, -67, -128, 2, -65, -128, -128, 127, -128, -126, 127, -41, -128,
    127, 58, 126, 75, -128, 127, -128, -63, -95, -116, 127, 15, -100, -120, 1, 119, 100, -14, 80,
    -128, -102, 119, -25, -119, 26, 63, 54, 127, -128, 5, -128, 127, -36, -128, -128, 127, 92, 127,
    -37, 127, 36, -63, 127, 127, -128, 115, 67, -128, 127, 127, 82, 84, -79, 127, -128, 127, -115,
    24, 127, 50, 89, -32, -128, 127, -16, 127, -15, 127, 68, -22, 127, 36, 100, 127, -53, -44, -54,
    -32, -5, -86, 127, 127, -3, 127, -128, 127, -126, 41, -96, 127, 82, -128, 127, -91, 17, -32,
    -128, 127, 61, 65, 111, 127, 127, -32, 127, -15, -30, 127, -127, 127, 127, -13, 33, 127, 74,
    127, 116, -128, 75, -128, -100, 127, 38, -128, -49, -128, -128, 127, -128, 127, 127, -6, -43,
    127, 127, 127, 127, -108, 115, 127, -82, 127, -128, 121, -21, 127, -128, -128, 127, -128, 127,
    127, 127, 127, 109, -128, -22, -97, 127, -128, -128, 127, 38, 92, 24, 76, -104, -128, 127, 39,
    -128, -128, -11, -128, 109, 86, -128, -64, 127, -128, 127, -128, -128, -91, 62, 127, 11, -128,
    -128, 87, 38, 117, 127, -128, 127, 127, 127, 127, 40, 104, 127, 113, 127, 127, -128, 53, -128,
    -26, -128, -128, 127, -128, -72, -128, 105, 127, -16, -124, 127, 5, -54, 24, 102, -128, 84, 74,
    51, 127, -128, 127, 109, 72, -128, -44, -102, -128, 127, -128, 127, 127, -25, 127, -128, -48,
    77, 60, 58, 38, 127, -128, 91, -6, 125, -128, 127, 127, -128, -58, -22, -127, -45, 127, -6,
    127, -128, 43, 127, 127, -128, -128, 81, 111, 19, 127, -128, 123, 60, -120, 127, -128, -128,
    -128, 127, -128, -128, 25, -128, -68, -52, -128, 127, 127, -71, 122, 119, -32, -128, -128,
    -128, -128, -29, -128, -27, 127, 127, 25, -128, 127, 73, 127, 127, 81, -3, -128, 125, -28, 118,
    -44, 127, -85, -98, -45, 127, 127, -128, -67, 127, -83, -9, 127, -128, -106, -21, 109, -128,
    -112, 1, -15, -76, -128, -37, -79, -128, 127, -128, -128, -7, 127, -128, -128, -128, 127, 127,
    -128, -118, 79, 127, -128, -128, -128, -16, 56, 127, 127, -128, 118, -128, -115, -128, 127,
    100, -128, -128, 92, -128, -128, 111, -128, 46, 127, -128, 83, 14, -128, -113, 10, 113, 127,
    127, -73, 127, -128, 127, -54, 93, 127, -27, -128, 127, -128, 127, -128, 24, -128, 93, 127, 99,
    127, -128, -128, 0, 29, 127, -64, 70, 127, -112, 92, 127, -116, 127, -8, -83, 127, -128, 28,
    127, 127, -53, -38, 127, 127, -77, 82, 127, -66, -121, 127, 127, 127, 127, 127, -103, -104, 75,
    127, -108, -128, 127, 39, -91, -50, 63, 36, -84, -60, -74, 127, -128, -128, 127, 127, 127, 127,
    127, 127, 127, -29, -128, -128, -128, -128, -128, -123, -128, -33, -128, -32, 127, 127, -128,
    -128, 29, -128, 28, -72, 127, 52, -11, -128, -128, 127, -103, 127, 6, 127, 116, -128, 45, 91,
    72, -127, -128, -128, -128, -113, -128, 127, 77, -128, -128, 25, -59, 7, 52, -30, -128, -52,
    -59, 27, 127, 125, -128, -9, -8, 72, -128, 127, 52, 96, 49, 105, -97, 127, -122, 127, -44, -13,
    -4, 127, -87, 127, 127, 38, 127, -71, -67, 127, 34, -128, -128, -1, -91, -128, 2, 84, 127,
    -128, -128, -128, 67, -102, -128, -78, -50, -128, 95, -128, -128, -128, -44, -115, 93, 127,
    -128, -60, -87, 71, 127, -42, 35, 126, -89, -128, -128, -78, -39, 127, 127, -109, -115, -2,
    -123, 120, -128, 118, -128, -128, -123, -20, -105, -128, -128, -128, -128, -128, -128, 0, -58,
    97, -128, 127, 91, -128, 127, -29, -128, 25, 120, 21, 107, 31, 127, 54, 46, 127, 48, 7, 127,
    -104, 91, -54, 127, -66, -128, -128, -128, -115, -112, -12, 127, -128, 127, -6, 74, -84, -40,
    127, -123, -126, 127, -38, -128, 20, -128, 127, -45, 127, -57, -128, 127, 127, -47, 127, 112,
    -34, 76, 127, -128, -128, 127, 127, -128, -71, 15, 127, 127, 122, -128, 80, -128, 41, 127,
    -123, -25, -22, -95, -110, 127, -128, -128, 98, 1, 4, -1, 127, 119, -128, -120, 127, -128,
    -128, 127, -70, 92, -53, 9, -123, -128, -128, -128, 84, 91, 127, 109, 127, -103, -128, 127,
    127, 82, 127, 127, 127, -51, -128, 94, -128, -128, -94, -128, 127
};

RT_LOCAL_DATA int8_t vec_exp_bias[] = {
    33, -114, 41, 13, -128, -128, -38, 103, 127, 51, 127, 16, -128, 21, -128, 53, 122, 127, -29,
    -95, 127, -117, -128, -1, 127, -128, 127, -18, -13, 127, 48, 127, -128, 115, 87, 113, -111,
    -128, 125, -4, 127, 103, 127, 119, -128, 69, -74, 127, 127, 95, 127, -128, 68, 127, -128, -128,
    127, -128, -128, 65, -128, 18, -128, 127, 127, -128, -96, 121, 66, 127, 127, 120, 78, -106,
    127, 39, -128, -65, 127, 127, 3, -113, 127, 127, 127, 65, 127, 127, 109, 32, 127, -128, -128,
    127, 36, 93, -31, 2, -128, -128, -128, 0, 127, 41, -88, 114, -128, -59, 127, 79, -128, 95,
    -128, 66, 101, -128, 127, -128, -128, 127, -86, -40, 127, -47, 127, 127, 127, 20, -128, -128,
    -121, -128, 55, 57, 105, 127, -128, 63, 72, -128, -101, -128, 127, 127, -128, 127, -128, 127,
    25, -128, 127, -23, -128, -128, 38, -128, 127, -124, 93, -57, 127, -55, 127, -128, 127, -71,
    -6, 127, 127, 127, 120, -124, -62, 52, 127, 127, 12, -128, -128, 21, -128, 127, -48, -128, 23,
    127, 37, -86, 18, -128, 93, -128, -128, -128, 120, -128, -128, -128, 85, 44, 127, -128, -122,
    -128, 113, 72, -128, 127, -128, -91, -128, -52, 127, 29, -128, -128, 127, -128, -128, 127, 127,
    -128, 77, 127, -128, -87, 127, 127, -67, 127, -81, 127, -45, 127, -69, -93, -128, 127, 29,
    -128, 127, -128, 127, -84, 127, 99, 98, -128, 127, 127, 116, 90, 127, 127, -18, 127, 81, -128,
    105, -128, -73, 51, 93, 110, 127, 127, -59, 127, 45, 64, -128, 127, -67, -92, -128, -128, -79,
    127, 127, 33, -128, -128, 127, -62, -128, 7, -60, -128, -128, 127, -128, -121, 127, -36, -128,
    127, 63, 127, 80, -128, 127, -128, -58, -90, -111, 127, 20, -95, -115, 6, 124, 105, -9, 85,
    -128, -97, 124, -20, -114, 31, 68, 59, 127, -128, 10, -128, 127, -31, -128, -128, 127, 97, 127,
    -32, 127, 41, -58, 127, 127, -128, 120, 72, -128, 127, 127, 87, 89, -74, 127, -128, 127, -110,
    29, 127, 55, 94, -27, -128, 127, -11, 127, -10, 127, 73, -17, 127, 41, 105, 127, -48, -39, -49,
    -27, 0, -81, 127, 127, 2, 127, -128, 127, -121, 46, -91, 127, 87, -128, 127, -86, 22, -27,
    -128, 127, 66, 70, 116, 127, 127, -27, 127, -10, -25, 127, -122, 127, 127, -8, 38, 127, 79,
    127, 121, -128, 80, -128, -95, 127, 43, -128, -44, -128, -128, 127, -128, 127, 127, -1, -38,
    127, 127, 127, 127, -103, 120, 127, -77, 127, -128, 126, -16, 127, -128, -128, 127, -128, 127,
    127, 127, 127, 114, -126, -17, -92, 127, -128, -128, 127, 43, 97, 29, 81, -99, -128, 127, 44,
    -128, -128, -6, -128, 114, 91, -128, -59, 127, -128, 127, -128, -128, -86, 67, 127, 16, -128,
    -128, 92, 43, 122, 127, -128, 127, 127, 127, 127, 45, 109, 127, 118, 127, 127, -128, 58, -128,
    -21, -128, -128, 127, -128, -67, -128, 110, 127, -11, -119, 127, 10, -49, 29, 107, -128, 89,
    79, 56, 127, -126, 127, 114, 77, -128, -39, -97, -128, 127, -128, 127, 127, -20, 127, -128,
    -43, 82, 65, 63, 43, 127, -128, 96, -1, 127, -128, 127, 127, -128, -53, -17, -122, -40, 127,
    -1, 127, -128, 48, 127, 127, -128, -128, 86, 116, 24, 127, -128, 127, 65, -115, 127, -128,
    -128, -128, 127, -128, -128, 30, -128, -63, -47, -128, 127, 127, -66, 127, 124, -27, -128,
    -128, -128, -128, -24, -128, -22, 127, 127, 30, -128, 127, 78, 127, 127, 86, 2, -128, 127, -23,
    123, -39, 127, -80, -93, -40, 127, 127, -128, -62, 127, -78, -4, 127, -128, -101, -16, 114,
    -128, -107, 6, -10, -71, -128, -32, -74, -123, 127, -128, -128, -2, 127, -128, -128, -128, 127,
    127, -128, -113, 84, 127, -128, -128, -128, -11, 61, 127, 127, -128, 123, -128, -110, -128,
    127, 105, -128, -128, 97, -128, -128, 116, -128, 51, 127, -128, 88, 19, -128, -108, 15, 118,
    127, 127, -68, 127, -128, 127, -49, 98, 127, -22, -128, 127, -128, 127, -128, 29, -128, 98,
    127, 104, 127, -128, -128, 5, 34, 127, -59, 75, 127, -107, 97, 127, -111, 127, -3, -78, 127,
    -128, 33, 127, 127, -48, -33, 127, 127, -72, 87, 127, -61, -116, 127, 127, 127, 127, 127, -98,
    -99, 80, 127, -103, -128, 127, 44, -86, -45, 68, 41, -79, -55, -69, 127, -128, -128, 127, 127,
    127, 127, 127, 127, 127, -24, -128, -128, -128, -128, -128, -118, -128, -28, -128, -27, 127,
    127, -128, -128, 34, -128, 33, -67, 127, 57, -6, -128, -128, 127, -98, 127, 11, 127, 121, -128,
    50, 96, 77, -122, -128, -127, -128, -108, -128, 127, 82, -128, -128, 30, -54, 12, 57, -25,
    -128, -47, -54, 32, 127, 127, -128, -4, -3, 77, -126, 127, 57, 101, 54, 110, -92, 127, -117,
    127, -39, -8, 1, 127, -82, 127, 127, 43, 127, -66, -62, 127, 39, -128, -127, 4, -86, -128, 7,
    89, 127, -128, -128, -128, 72, -97, -128, -73, -45, -128, 100, -128, -128, -128, -39, -110, 98,
    127, -128, -55, -82, 76, 127, -37, 40, 127, -84, -128, -128, -73, -34, 127, 127, -104, -110, 3,
    -118, 125, -128, 123, -126, -128, -118, -15, -100, -128, -128, -128, -128, -128, -128, 5, -53,
    102, -128, 127, 96, -128, 127, -24, -128, 30, 125, 26, 112, 36, 127, 59, 51, 127, 53, 12, 127,
    -99, 96, -49, 127, -61, -128, -128, -128, -110, -107, -7, 127, -128, 127, -1, 79, -79, -35,
    127, -118, -121, 127, -33, -128, 25, -128, 127, -40, 127, -52, -128, 127, 127, -42, 127, 117,
    -29, 81, 127, -128, -128, 127, 127, -128, -66, 20, 127, 127, 127, -128, 85, -128, 46, 127,
    -118, -20, -17, -90, -105, 127, -128, -128, 103, 6, 9, 4, 127, 124, -128, -115, 127, -128,
    -128, 127, -65, 97, -48, 14, -118, -128, -128, -128, 89, 96, 127, 114, 127, -98, -128, 127,
    127, 87, 127, 127, 127, -46, -128, 99, -128, -127, -89, -128, 127
};

#endif//__TEST_STIMULI_H__