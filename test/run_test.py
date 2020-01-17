#! /usr/bin/python3

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

# setup environmental variables
os.environ["EEGNET_WOLF_SRC_PATH"] = os.path.realpath(os.path.join(os.getcwd(), "../src"))
os.environ["PYTHONPATH"] += ":" + os.path.realpath(os.path.join(os.getcwd(), "../python_utils"))

old_cwd = os.getcwd()

# keep track of statistics
num_total = 0
num_success = 0

# search recursively all files
for root, _, files in os.walk("."):
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
