"""
These Utility functions enable easy unit tests for the EEGnet.
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.0"
__date__ = "2020/01/19"


def parse_output(filename):
    """
    This function parses the output of a test run.

    For each run of the test, the program should print the following:
        ## ID: result: [OK|FAIL]
        ## ID: cycles: N_CYCLES
        ## ID: instructions: N_INSTR

    Multiple runs are allowed.
    Make sure, that you pipe the execution into a file, whose name is then passed into this function.

    Example:
        os.system("make clean all run > result.out")
        parsed = parse_output("result.out")

    This function returns a dictionary in the form:
        { "1": {"result": "OK", "cycles": 215, "instructions": 201}, ... }
    """
    parsed = {}
    with open(filename, "r") as _f:
        for line in _f.readlines():
            if not line.startswith("## "):
                continue
            line = line.lstrip("# ")
            parts = [p.strip() for p in line.split(":")]
            assert len(parts) == 3, line
            if parts[0] not in parsed:
                parsed[parts[0]] = {}
            if parts[1] == "result":
                parsed[parts[0]]["result"] = parts[2] == "OK"
            if parts[1] == "cycles":
                parsed[parts[0]]["cycles"] = int(parts[2])
            if parts[1] == "instructions":
                parsed[parts[0]]["instructions"] = int(parts[2])
    return parsed


class TestLogger:
    """
    Class to display the logging result
    """
    def __init__(self, name):
        self.name = name
        self.num_cases = 0
        self.num_successful = 0
        print("\n**** Test Case: {}".format(self.name))

    def show_subcase_result(self, subcase_name, results):
        """
        Display the parsed results and count the number of (successful) test cases.

        Parameters:
        - subcase_name: str, name of the subcase to be displayed
        - results: parsed results file
        """
        assert results
        if len(results) == 1:
            result = list(results.values())[0]
            success_str = "Ok" if result["result"] else "FAIL"
            print("{}: {} (cycles: {}, instructions: {})"
                  .format(subcase_name, success_str, result["cycles"], result["instructions"]))

            # keep track of statistics
            self.num_cases += 1
            if result["result"]:
                self.num_successful += 1
        else:
            for case_id, result in results.items():
                success_str = "Ok" if result["result"] else "FAIL"
                print("{} [{}]: {} (cycles: {}, instructions: {})"
                    .format(subcase_name, case_id, success_str, result["cycles"], result["instructions"]))

                # keep track of statistics
                self.num_cases += 1
                if result["result"]:
                    self.num_successful += 1

    def summary(self):
        """
        Returns tuple: (number of test cases, number of successful test cases)
        """
        return self.num_cases, self.num_successful
