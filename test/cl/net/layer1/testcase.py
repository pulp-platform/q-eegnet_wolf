"""
This file will test the convolution implementation
"""

import random
import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray, align_array
from makefile import Makefile
from golden_model import GoldenModel
import functional as F

TESTNAME = "cl::net::layers"
RESULT_FILE = "result.out"

INPUT_FILENAME = "../../../../data/input.npz"
NET_FILENAME = "../../../../data/net.npz"
CONFIG_FILENAME = "../../../../data/config.json"


def gen_stimuli(random_input):
    """
    This function generates the stimuli (input and output) for the test
    """
    model = GoldenModel(CONFIG_FILENAME, NET_FILENAME, clip_balanced=False)
    if random_input:
        x = np.random.randint(-60, 60, (model.C, model.T))
    else:
        x = np.load(INPUT_FILENAME)["input"][0, :, :]
        x = F.quantize_to_int(x, model.input_scale)
    y_exp = model.layers[0](x)
    return x, y_exp


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for intrinsic_conv_scale in [False, True]:
        for simd in [False, True]:
            for parallel in [False, True]:
                for cross_correlate in [False, True]:

                    if not simd and (parallel or cross_correlate):
                        continue

                    # parallel requires intrinsic conv scale
                    if parallel and not intrinsic_conv_scale:
                        continue

                    # not implemented
                    if cross_correlate and not parallel:
                        continue

                    # generate makefile
                    mkf = Makefile()
                    mkf.add_fc_test_source("test.c")
                    mkf.add_cl_test_source("cluster.c")
                    mkf.add_cl_prog_source("net/layer1.c")
                    mkf.add_cl_prog_source("net/net.c")
                    mkf.add_cl_prog_source("func/conv.c")
                    mkf.add_cl_prog_source("func/xcorr.c")
                    mkf.add_cl_prog_source("func/transform.c")

                    if parallel:
                        mkf.add_define("PARALLEL")
                    if intrinsic_conv_scale:
                        mkf.add_define("INTRINSIC_SCALE")
                    if cross_correlate:
                        mkf.add_define("CROSS_CORRELATE")
                    if not simd:
                        mkf.add_define("NO_SIMD")

                    mkf.write()

                    random_input = False

                    # generate the stimuli
                    x, y_exp = gen_stimuli(random_input)
                    x_align = align_array(x)
                    y_exp_align = align_array(y_exp)

                    # prepare header file
                    header = HeaderFile("test_stimuli.h")
                    header.add(HeaderArray("x_vec", "int8_t", x_align.ravel()))
                    header.add(HeaderArray("y_exp_vec", "int8_t", y_exp_align.ravel()))
                    header.write()

                    # compile and run
                    os.system("make clean all run > {}".format(RESULT_FILE))

                    # parse output
                    result = parse_output(RESULT_FILE)

                    # log the result
                    options = []
                    if simd:
                        options.append("simd")
                    if parallel:
                        options.append("par")
                    if intrinsic_conv_scale:
                        options.append("intr.s.")
                    if cross_correlate:
                        options.append("xcorr")

                    subcase_name = "Layer 1 "
                    if options:
                        subcase_name += "; ".join(options)
                    else:
                        subcase_name += "naive"

                    logger.show_subcase_result(subcase_name, result)

    # return summary
    return logger.summary()
