import os
import json
import numpy as np
from tqdm import tqdm

from data.gen_input_header import gen_input_header
from data.gen_net_header import gen_net_header

# generate the header
gen_net_header("data/net.npz", "data/config.json", "src/cl/net/net.h")

# load network
net = np.load("data/net.npz")
# load verification data
data = dict(np.load("data/benchmark.npz"))

# load configuration file
with open("data/config.json", "r") as _f:
    config = json.load(_f)
# we only need the network parameters
net_params = config["indiv"]["net"]["params"]

num_samples = 0
num_correct = 0
num_pred_equal = 0

for sample, label, exp_pred in tqdm(zip(data["samples"], data["labels"], data["predictions"]),
                                    total=data["samples"].shape[0]):

    # generate input header
    gen_input_header(net, net_params, sample[0], "src/cl/input.h")

    # run program and write output to file
    os.system("make clean all run > result.out")

    # parse result.out
    classes = np.zeros(4)
    with open("result.out", "r") as result_file:
        for line in result_file:
            line = line.strip()
            if line.startswith("Class "):
                parts = line.split(":")
                assert len(parts) == 2
                assert len(parts[0].split(" ")) == 2
                class_id = int(parts[0].split(" ")[1]) - 1
                result = int(parts[1])
                classes[class_id] = result

    # get the maximum class
    acq_pred = np.argmax(classes)
    exp_pred = np.argmax(exp_pred)

    if acq_pred != exp_pred:
        num_pred_equal += 1

    if acq_pred == label:
        num_correct += 1

    num_samples += 1

print("Result: accuracy = {}%, similarity = {}%".format(100 * num_correct / num_samples,
                                                        100 * num_pred_equal / num_samples))
