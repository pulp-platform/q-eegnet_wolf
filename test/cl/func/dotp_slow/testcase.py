"""
This file will test the convolution implementation
"""

import os
import numpy as np

from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray
from makefile import Makefile
import functional as F

TESTNAME = "cl::func::dotp_slow"
RESULT_FILE = "result.out"

def gen_stimuli(length, a_stride=8, b_stride=8):
    """
    This function generates the stimuli (input and output) for the test
    """
    vec_a = np.random.randint(-128, 127, (length, a_stride)).astype(int)
    vec_b = np.random.randint(-128, 127, (length, b_stride)).astype(int)
    result = np.dot(vec_a[:, 0], vec_b[:, 0])
    return vec_a, vec_b, result


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    # generate makefile
    mkf = Makefile()
    mkf.add_fc_test_source("test.c")
    mkf.add_cl_test_source("cluster.c")
    mkf.add_cl_prog_source("func/dotp.c")
    mkf.write()

    for length in [22, 23, 1024, 1025]:
        for a_stride, b_stride in [(1, 1), (4, 1), (8, 4)]:
            # generate the stimuli
            vec_a, vec_b, exp_result = gen_stimuli(length, a_stride, b_stride)

            # prepare header file
            header = HeaderFile("test_stimuli.h")
            header.add(HeaderConstant("LENGTH", length))
            header.add(HeaderConstant("EXP_RESULT", exp_result))
            header.add(HeaderConstant("A_STRIDE", a_stride))
            header.add(HeaderConstant("B_STRIDE", b_stride))
            header.add(HeaderArray("vec_a", "int8_t", vec_a.ravel()))
            header.add(HeaderArray("vec_b", "int8_t", vec_b.ravel()))
            header.write()

            # compile and run
            os.system("make clean all run > {}".format(RESULT_FILE))

            # parse output
            result = parse_output(RESULT_FILE)

            # log the result
            subcase_name = "length: {}, stride: {}x{}".format(length, a_stride, b_stride)
            logger.show_subcase_result(subcase_name, result)

    # return summary
    return logger.summary()
