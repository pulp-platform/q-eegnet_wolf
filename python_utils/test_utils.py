def parse_output(filename):
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
        print("*** Test Case: {}".format(self.name))

    def show_subcase_result(self, subcase_name, results):
        assert results
        if len(results) == 1:
            result = list(results.values())[0]
            success_str = "Ok" if result["result"] else "FAIL"
            print("{}: {} (cycles: {}, instructions: {})"
                  .format(subcase_name, success_str, result["cycles"], result["instructions"]))
        else:
            for case_id, result in results.items():
                success_str = "Ok" if result["result"] else "FAIL"
                print("{} [{}]: {} (cycles: {}, instructions: {})"
                    .format(subcase_name, case_id, success_str, result["cycles"], result["instructions"]))
        return all([r["result"] for r in results.values()])
