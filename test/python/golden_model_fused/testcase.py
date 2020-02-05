"""
Test the golden model
"""

import numpy as np
from test_utils import TestLogger
from golden_model import GoldenModel, Layer1, Layer2, Layer3, Layer4, Layer5, FusedLayer12
import convert_torch_format as convert
import functional as F

# TODO add a meaningful name for the test
TESTNAME = "python::GoldenModel with fused layer 1 and 2"
NET_FILENAME = "../../../data/net.npz"
DATA_FILENAME = "../../../data/verification.npz"
CONFIG_FILENAME = "../../../data/config.json"

TOLERANCE = 6


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """
    logger = TestLogger(TESTNAME)
    
    # load net and the data
    model = GoldenModel(CONFIG_FILENAME, NET_FILENAME, no_scale_between_l1_l2=True)
    data = dict(np.load(DATA_FILENAME))
    net = dict(np.load(NET_FILENAME))

    for i, layer in enumerate(model.layers):
        result = test_layer(layer, data)
        test_index = i + 2
        if i == 0:
            test_index = "1+2"
        logger.show_subcase_result("Layer {}".format(test_index), result)


    result = test_model(model, data)
    logger.show_subcase_result("Model", result)

    # return summary
    return logger.summary()


def test_model(model, data):
    """
    Test the entire Golden Model
    """
    assert isinstance(model, GoldenModel)

    x = data["input"]
    x = np.reshape(x, model.input_shape)
    x = F.quantize_to_int(x, model.input_scale)

    test_names = ["layer2_activ", "layer3_activ", "layer4_activ", "output_quant"]

    result = {}

    for i, layer, test_name in zip(range(1, 5), model.layers, test_names):
        y_exp = data[test_name]
        y_exp = np.reshape(y_exp, layer.output_shape)
        y_exp = F.quantize_to_int(y_exp, layer.output_scale)

        x = layer(x)

        test_index = i + 1
        if i == 1:
            test_index = "1+2"

        result.update(_compare_result(x, y_exp, test_index))

    return result


def test_layer(layer, data):
    """
    Test a specific layer
    """
    if isinstance(layer, FusedLayer12):
        x = data["input"]
        y = data["layer2_activ"]
    elif isinstance(layer, Layer3):
        x = data["layer2_activ"]
        y = data["layer3_activ"]
    elif isinstance(layer, Layer4):
        x = data["layer3_activ"]
        y = data["layer4_activ"]
    elif isinstance(layer, Layer5):
        x = data["layer4_activ"]
        y = data["output_quant"]
    else:
        raise TypeError("Argument layer must be a Layer")

    x = np.reshape(x, layer.input_shape)
    y = np.reshape(y, layer.output_shape)
    x = F.quantize_to_int(x, layer.input_scale)
    y = F.quantize_to_int(y, layer.output_scale)
    
    y_hat = layer(x)

    return _compare_result(y_hat, y)


def _compare_result(y_exp, y_hat, test_index=1, tolerance=10, epsilon=1e-4):
    """
    The error is computed in the following way:
    1. Scale the acquired output y_hat back into regular floating point representation
    2. take the maximal absolute difference e = max(|y_hat - y_exp|)
    3. Scale the error such that 1 represents an entire quantization step.
    4. We consider the output to be correct if this maximal scaled error is less than 1

    Parameters:
    - y_exp: np.array(type=int), expected result in integer representation
    - y_hat: np.array(type=int), acquired result in integer representation
    - test_index: index used in the return dict
    - tolerance: number of quantization steps the output is allowed to differ from the expected one.
    - epsilon: numerical stability

    Returns: dictionary: {test_index: {"result": bool, "max_error": int, "mean_error": float}}
    """
    abs_diff = np.abs(y_hat - y_exp)
    max_err = np.max(abs_diff)
    mean_err = np.mean(abs_diff)

    are_equal = max_err <= (tolerance + epsilon)

    return {str(test_index): {"result": are_equal, "max error": int(round(max_err)),
                              "avg error": mean_err}}
