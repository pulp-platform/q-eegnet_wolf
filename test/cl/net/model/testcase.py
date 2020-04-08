"""
This file will test the convolution implementation
"""

import random
import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray, align_array, align_array_size
from makefile import Makefile
from golden_model import GoldenModel
import functional as F

TESTNAME = "cl::net::model"
RESULT_FILE = "result.out"

INPUT_FILENAME = "../../../../data/verification.npz"
NET_FILENAME = "../../../../data/net.npz"
CONFIG_FILENAME = "../../../../data/config.json"


def gen_stimuli(random_input=False, no_div=False, pad_data=False, reorder_bn=True):
    """
    This function generates the stimuli (input and output) for the test
    """
    model = GoldenModel(CONFIG_FILENAME, NET_FILENAME, clip_balanced=False, no_scale_between_l1_l2=no_div, reorder_bn=reorder_bn)
    if random_input:
        x = np.random.randint(-60, 60, (model.C, model.T))
    else:
        x = np.load(INPUT_FILENAME)["input"][0, :, :]
        x = F.quantize_to_int(x, model.input_scale)
    y_exp = model(x)
    y_exp_align = align_array(y_exp)

    if pad_data:
        C, T = x.shape
        T_pad = T + 63
        assert T_pad % 4 == 0
        x_pad = np.zeros((C, T_pad), dtype=np.int)
        x_pad[:, 31:31 + T] = x
        return x, x_pad, y_exp, y_exp_align
    else:
        x_align = align_array(x)
        return x, x_align, y_exp, y_exp_align


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for intrinsic, simd, flip_layers, parallel, stream, xcorr, fuse, no_div, reorder, dup_inp in [
            (False, False, False, False, False, False, False, False, False, False),
            (True, False, False, False, False, False, False, False, False, False),
            (True, True, False, False, False, False, False, False, False, False),
            (True, True, True, False, False, False, False, False, False, False),
            (True, True, True, True, False, False, False, False, False, False),
            (True, True, True, True, True, False, False, False, False, False),
            (True, True, True, True, True, True, False, False, False, False),
            (True, True, True, True, True, True, True, False, False, False),
            (True, True, True, True, True, True, True, True, False, False),
            (True, True, True, True, True, True, True, True, True, False),
            (True, True, True, True, True, True, True, True, True, True)
    ]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("net/model.c")
        mkf.add_cl_prog_source("net/layer1.c")
        mkf.add_cl_prog_source("net/layer2.c")
        mkf.add_cl_prog_source("net/layer3.c")
        mkf.add_cl_prog_source("net/layer4.c")
        mkf.add_cl_prog_source("net/layer5.c")
        mkf.add_cl_prog_source("net/fused_layer_1_2.c")
        mkf.add_cl_prog_source("net/net.c")
        mkf.add_cl_prog_source("func/transform.c")
        mkf.add_cl_prog_source("func/dotp.c")
        mkf.add_cl_prog_source("func/conv.c")
        mkf.add_cl_prog_source("func/flip.c")
        mkf.add_cl_prog_source("func/xcorr.c")

        if not simd:
            mkf.add_define("NO_SIMD")
        if flip_layers:
            mkf.add_define("FLIP_LAYERS")
        if parallel:
            mkf.add_define("PARALLEL")
        if intrinsic:
            mkf.add_define("INTRINSIC_SCALE")
        if stream:
            mkf.add_define("DMA_STREAM")
        if xcorr:
            mkf.add_define("CROSS_CORRELATE")
        if fuse:
            mkf.add_define("FUSE_LAYERS")
        if no_div:
            mkf.add_define("NO_INTERMEDIATE_SCALE")
        if dup_inp:
            mkf.add_define("DUPLICATE_FEATUREMAP")
        if reorder:
            mkf.add_define("REORDER_BN")

        mkf.write()

        # generate the stimuli
        _, x_align, _, y_exp_align = gen_stimuli(no_div=no_div, pad_data=dup_inp, reorder_bn=reorder)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        header.add(HeaderArray("x_vec", "int8_t", x_align.ravel()))
        header.add(HeaderArray("y_exp_vec", "int8_t", y_exp_align.ravel()))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        # skip the naive result
        if not flip_layers:
            result["1"]["result"] = None

        # prepare the case name
        subcase_name = "naive"
        if intrinsic:
            subcase_name = "+ intrinsic scale"
        if simd:
            subcase_name = "+ SIMD"
        if flip_layers:
            subcase_name = "+ flip"
        if parallel:
            subcase_name = "+ parallel"
        if stream:
            subcase_name = "+ double buffering"
        if xcorr:
            subcase_name = "+ cross correlations"
        if fuse:
            subcase_name = "+ fused layer 1+2"
        if no_div:
            subcase_name = "+ no division after layer 1"
        if reorder:
            subcase_name = "+ reorder BN"
        if dup_inp:
            subcase_name = "+ duplicate featuremap"

        # log the result
        logger.show_subcase_result(subcase_name, result)

    # return summary
    return logger.summary()
