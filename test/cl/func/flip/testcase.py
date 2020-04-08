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
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderScalar
from header_file import align_array, align_array_size
from makefile import Makefile
import functional as F

TESTNAME = "cl::func::flip"
RESULT_FILE = "result.out"

SCALE_FACTOR = 10
BIAS = 50

def gen_stimuli(outer_len, inner_len):
    """
    This function generates the stimuli (input and output) for the test
    """
    inp = np.random.randint(-128, 127, (outer_len, inner_len))
    oup = np.transpose(inp)
    inp = align_array(inp)
    oup = align_array(oup)
    return inp.ravel(), oup.ravel()


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
    mkf.add_cl_prog_source("func/flip.c")
    mkf.write()

    for outer_len in [16, 17, 18, 19]:
        for inner_len in [256, 257, 258, 259]:
            # generate the stimuli
            stim, exp = gen_stimuli(outer_len, inner_len)

            # prepare header file
            header = HeaderFile("test_stimuli.h")
            header.add(HeaderConstant("OUTER_LEN", outer_len))
            header.add(HeaderConstant("OUTER_LEN_ALIGN", align_array_size(outer_len)))
            header.add(HeaderConstant("INNER_LEN", inner_len))
            header.add(HeaderConstant("INNER_LEN_ALIGN", align_array_size(inner_len)))
            header.add(HeaderArray("vec_x", "int8_t", stim))
            header.add(HeaderArray("vec_exp", "int8_t", exp))
            header.write()

            # compile and run
            os.system("make clean all run > {}".format(RESULT_FILE))

            # parse output
            result = parse_output(RESULT_FILE)

            # log the result
            subcase_name = "{}x{}".format(outer_len, inner_len)
            logger.show_subcase_result(subcase_name, result)

    # return summary
    return logger.summary()
