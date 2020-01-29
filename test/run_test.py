"""
This script searches recursively the current directory and executes all "gen_test.py" files.
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.0"
__date__ = "2020/01/19"

import os
import sys
import importlib.util

TEST_FILENAME = "testcase.py"


def test_main(root_folder):
    """ main function """

    old_cwd = os.getcwd()

    # go a directory up and build the project, without running it
    os.chdir("..")
    print("Building the project...")
    os.system("./run.sh -b > /dev/null")
    # go back to the test directory
    os.chdir(old_cwd)

    # keep track of statistics
    num_total = 0
    num_success = 0

    # search recursively all files
    for root, _, files in os.walk(root_folder):
        for filename in files:
            if filename == TEST_FILENAME:
                # import the file
                spec = importlib.util.spec_from_file_location("testcase", os.path.join(root, filename))
                testcase = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(testcase)

                # change directory into the test directory
                os.chdir(root)

                # execute the test
                _num_total, _num_success = testcase.test()

                num_total += _num_total
                num_success += _num_success

                # leave directory again
                os.chdir(old_cwd)

    # print a summary
    print("\n********************")
    print("Summary: {}".format("OK" if num_total == num_success else "FAIL"))
    print("Number of tests: {}".format(num_total))
    print("Failed tests:    {}".format(num_total - num_success))


if __name__ == "__main__":
    ROOT_FOLDER = "."
    if len(sys.argv) > 1:
        ROOT_FOLDER = sys.argv[1]

    test_main(ROOT_FOLDER)
