"""
TODO Explain what this test will verify
"""

import random
import os
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray

# TODO add a meaningful name for the test
TESTNAME = "example"
RESULT_FILE = "result.out"

# TODO implement this function
def gen_stimuli(size=1024):
    """
    This function generates the stimuli (input and output) for the test
    """
    vecA = [random.randint(-128, 127) for _ in range(size)]
    vecB = [random.randint(-128, 127) for _ in range(size)]
    result = sum([a * b for a, b in zip(vecA, vecB)])
    return vecA, vecB, result

def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """
    # TODO remove this line
    return 0, 0

    logger = TestLogger(TESTNAME)

    # TODO change this loop to do more suitable tests
    for size in [128, 1024, 4096]:
        # generate the stimuli
        vecA, vecB, result = gen_stimuli(size)

        # prepare header file
        # TODO generate the header file
        header = HeaderFile("test_stimuli.h")
        header.add(HeaderConstant("LENGTH", size))
        header.add(HeaderArray("vecA", "int8_t", vecA))
        header.add(HeaderArray("vecB", "int8_t", vecB))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        # log the result
        # TODO add meaningful name for the subcase
        logger.show_subcase_result("size {:4}".format(size), result)

    # return summary
    return logger.summary()
