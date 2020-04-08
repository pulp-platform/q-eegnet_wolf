"""
This file will test the convolution implementation
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "1.0"
__license__ = "Apache 2.0"
__copyright__ = """
    Copyright (C) 2020 ETH Zurich. All rights reserved.

    Author: Tibor Schneider, ETH Zurich

    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the License); you may
    not use this file except in compliance with the License.
    You may obtain a copy of the License at

    www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an AS IS BASIS, WITHOUT
    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


import random
import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray
from makefile import Makefile

TESTNAME = "cl::func::conv_scale"
RESULT_FILE = "result.out"


def gen_stimuli(size_a, size_b, scale, offset):
    """
    This function generates the stimuli (input and output) for the test
    """
    vecA = [random.randint(-128, 127) for _ in range(size_a)]
    vecB = [random.randint(-128, 127) for _ in range(size_b)]
    result = np.convolve(vecA, vecB, mode="valid")
    result = (result + offset) / scale
    result = result.astype(int)
    result = np.clip(result, -128, 127)
    return vecA, vecB, result


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for size_a, size_b in [(155, 16), (1188, 64), (4096, 128)]:
        for conv_version in [0, 1, 2, 3]:

            div_factor = 128 * size_b // 8
            offset = 10 * div_factor

            # generate makefile
            mkf = Makefile()
            mkf.add_fc_test_source("test.c")
            mkf.add_cl_test_source("cluster.c")
            mkf.add_cl_prog_source("func/conv.c")
            mkf.add_define("CONV_VERSION", conv_version)
            mkf.write()

            # generate the stimuli
            vecA, vecB, vecExp = gen_stimuli(size_a, size_b, div_factor, offset)

            # prepare header file
            header = HeaderFile("test_stimuli.h")
            header.add(HeaderConstant("LENGTH_A", size_a))
            header.add(HeaderConstant("LENGTH_B", size_b))
            header.add(HeaderConstant("LENGTH_RES", len(vecExp)))
            header.add(HeaderConstant("FACTOR", div_factor))
            header.add(HeaderConstant("OFFSET", offset))
            header.add(HeaderArray("vecA", "int8_t", vecA))
            header.add(HeaderArray("vecB", "int8_t", vecB))
            header.add(HeaderArray("vecExp", "int8_t", vecExp))
            header.write()

            # compile and run
            os.system("make clean all run > {}".format(RESULT_FILE))

            # parse output
            result = parse_output(RESULT_FILE)

            casename = "V{}, {}x{}".format(conv_version, size_a, size_b)

            # log the result
            logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
