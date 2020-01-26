"""
Class to geerate C Header Files
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.1"
__date__ = "2020/01/28"

import re
from textwrap import wrap
import numpy as np

MAX_WIDTH = 100
TAB = "    "

class HeaderFile():
    """
    Enables comfortable generation of header files
    """
    def __init__(self, filename, define_guard=None):
        self.filename = filename
        self.define_guard = define_guard
        if self.define_guard is None:
            self.define_guard = "__" + re.sub("[./]", "_", filename.upper()) + "__"
        self.elements = []

    def add(self, element):
        self.elements.append(element)

    def __str__(self):
        ret = ""
        ret += "#ifndef {}\n".format(self.define_guard)
        ret += "#define {}\n\n".format(self.define_guard)
        ret += "#include \"rt/rt_api.h\"\n\n"

        for element in self.elements:
            ret += str(element)

        ret += "#endif//{}".format(self.define_guard)
        return ret

    def write(self):
        with open(self.filename, "w") as _f:
            _f.write(str(self))


class HeaderEntry():
    def __init__(self):
        pass

    def __str__(self):
        return ""


class HeaderConstant(HeaderEntry):
    def __init__(self, name, value, blank_line=True):
        self.name = name
        self.value = value
        self.blank_line = blank_line

    def __str__(self):
        ret = "#define {} {}\n".format(self.name, self.value);
        if self.blank_line:
            ret += "\n"
        return ret


class HeaderScalar(HeaderEntry):
    def __init__(self, name, dtype, value, blank_line=True):
        self.name = name
        self.dtype = dtype
        self.value = value
        self.blank_line = blank_line

    def __str__(self):
        ret = "{} {} = {};\n".format(self.dtype, self.name, self.value)
        if self.blank_line:
            ret += "\n"
        return ret


class HeaderArray(HeaderEntry):
    def __init__(self, name, dtype, data, locality="RT_LOCAL_DATA", blank_line=True):
        assert locality in ["RT_LOCAL_DATA", "RT_L2_DATA", "RT_CL_DATA"]
        self.name = name
        self.dtype = dtype
        self.data = data
        self.locality = locality
        self.blank_line = blank_line

    def __str__(self):
        # first, try it as a one-liner
        ret = "{} {} {}[] = {{ {} }};".format(self.locality, self.dtype, self.name,
                                              ", ".join([str(item) for item in self.data]))
        if len(ret) <= MAX_WIDTH:
            ret += "\n"
            if self.blank_line:
                ret += "\n"
            return ret

        # It did not work on one line. Make it multiple lines
        ret = ""
        ret += "{} {} {}[] = {{\n".format(self.locality, self.dtype, self.name)
        line = "{}".format(TAB)
        for item in self.data:
            item_str = "{}, ".format(item)
            if len(line) + len(item_str) > MAX_WIDTH:
                ret += line.rstrip() + "\n"
                line = "{}".format(TAB)
            line += item_str
        ret += line.rstrip(", ") + "\n"
        ret += "};\n"
        if self.blank_line:
            ret += "\n"
        return ret


class HeaderComment(HeaderEntry):
    def __init__(self, text, mode="//", blank_line=True):
        assert mode in ["//", "/*"]
        self.text = text
        self.mode = mode
        self.blank_line = blank_line

    def __str__(self):
        if self.mode == "/":
            start = "// "
            mid = "\n// "
            end = ""
        else:
            start = "/*\n * "
            mid = "\n * "
            end = "\n */"
        ret = start
        ret += mid.join([mid.join(wrap(par, MAX_WIDTH-3)) for par in self.text.split("\n")])
        ret += end
        ret += "\n"
        if self.blank_line:
            ret += "\n"
        return ret


def align_array(x, n=4, fill=0):
    """
    Aligns the array to n elements (i.e. Bytes) by inserting fill bytes

    Parameters:
    - x: np.array(shape: [..., D])
    - n: number of elements to be aligned with
    - fill: value of the added elements

    Return: np.array(shape: [..., D']), where D' = ceil(D / n) * n
    """

    new_shape = (*x.shape[:-1], align_array_size(x.shape[-1]))
    y = np.zeros(new_shape, dtype=x.dtype)

    original_slice = tuple(slice(0, d) for d in x.shape)
    fill_slice = (*tuple(slice(0, d) for d in x.shape[:-1]), slice(x.shape[-1], y.shape[-1]))

    y[original_slice] = x
    y[fill_slice] = fill

    return y


def align_array_size(D, n=4):
    """
    Computes the required dimension D' such that D' is aligned and D <= D'

    Parameters:
    - D: Dimension to be aligned
    - n: number of elements to be aligned with

    Return: D'
    """
    return int(np.ceil(D / n) * n)
