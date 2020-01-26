"""
This file will test the convolution implementation
"""

import random
import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderScalar
import functional as F

TESTNAME = "cl::func::transform (with rounding)"
RESULT_FILE = "result.out"

SCALE_FACTOR = 10
BIAS = 50

def gen_stimuli(size = 1024, scale_factor=10, bias=50, max_val=2560):
    """
    This function generates the stimuli (input and output) for the test
    """
    x = [random.randint(-max_val, max_val) for _ in range(size)]
    y = list(F.apply_factor_offset(np.array(x), scale_factor, round=True, clip_balanced=False))
    y_bias = list(F.apply_factor_offset(np.array(x), scale_factor, bias, round=True, clip_balanced=False))
    return x, y, y_bias


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for size in [1024, 1025, 1026, 1027]:
        # generate the stimuli
        x, y, y_bias = gen_stimuli(size, scale_factor=SCALE_FACTOR, bias=BIAS, max_val=2560)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        header.add(HeaderConstant("LENGTH", size))
        header.add(HeaderScalar("div_factor", "int32_t", SCALE_FACTOR))
        header.add(HeaderScalar("bias", "int32_t", BIAS))
        header.add(HeaderArray("vec_x", "int32_t", x))
        header.add(HeaderArray("vec_exp", "int8_t", y))
        header.add(HeaderArray("vec_exp_bias", "int8_t", y_bias))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        # log the result
        logger.show_subcase_result("n={}".format(size), result)

    # return summary
    return logger.summary()
