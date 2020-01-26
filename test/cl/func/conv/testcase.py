"""
This file will test the convolution implementation
"""

import random
import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray

TESTNAME = "cl::func::conv"
RESULT_FILE = "result.out"


def gen_stimuli(size_a = 1125, size_b = 64):
    """
    This function generates the stimuli (input and output) for the test
    """
    vecA = [random.randint(-128, 127) for _ in range(size_a)]
    vecB = [random.randint(-128, 127) for _ in range(size_b)]
    result = list(np.convolve(vecA, vecB, mode="valid"))
    return vecA, vecB, result


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for size_a, size_b in [(155, 16), (1188, 64), (4096, 128)]:
        # generate the stimuli
        vecA, vecB, vecExp = gen_stimuli(size_a, size_b)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        header.add(HeaderConstant("LENGTH_A", size_a))
        header.add(HeaderConstant("LENGTH_B", size_b))
        header.add(HeaderConstant("LENGTH_RES", len(vecExp)))
        header.add(HeaderArray("vecA", "int8_t", vecA))
        header.add(HeaderArray("vecB", "int8_t", vecB))
        header.add(HeaderArray("vecExp", "int32_t", vecExp))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        # log the result
        logger.show_subcase_result("x".join([str(size_a), str(size_b)]), result)

    # return summary
    return logger.summary()
