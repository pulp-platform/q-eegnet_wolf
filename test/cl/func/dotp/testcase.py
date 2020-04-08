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


import os
import numpy as np

from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray
from makefile import Makefile
import functional as F

TESTNAME = "cl::func::dotp"
RESULT_FILE = "result.out"

def gen_stimuli(length):
    """
    This function generates the stimuli (input and output) for the test
    """
    vec_a = np.random.randint(-128, 127, (length, )).astype(int)
    vec_b = np.random.randint(-128, 127, (length, )).astype(int)
    result = np.dot(vec_a, vec_b)
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

    for length in [22, 24, 1024, 1025, 1026, 1027, 1028, 1029 ,1030, 1031]:
        # generate the stimuli
        vec_a, vec_b, exp_result = gen_stimuli(length)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        header.add(HeaderConstant("LENGTH", length))
        header.add(HeaderConstant("EXP_RESULT", exp_result))
        header.add(HeaderArray("vec_a", "int8_t", vec_a))
        header.add(HeaderArray("vec_b", "int8_t", vec_b))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        # log the result
        subcase_name = "length: {}".format(length)
        logger.show_subcase_result(subcase_name, result)

    # return summary
    return logger.summary()
