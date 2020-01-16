import re

MAX_WIDTH = 100
TAB = "    "

class HeaderFile():
    """
    Enables comfortable generation of header files
    """
    def __init__(self, filename):
        self.filename = filename
        self.define_guard = "__" + re.sub("[./]", "_", filename.upper()) + "__"
        self.elements = []

    def add(self, element):
        self.elements.append(element)

    def __str__(self):
        ret = ""
        ret += "#ifndef {}\n".format(self.define_guard)
        ret += "#define {}\n\n".format(self.define_guard)
        ret += "#include \"stdint.h\"\n\n"

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
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return "#define {} {}\n\n".format(self.name, self.value);


class HeaderScalar(HeaderEntry):
    def __init__(self, name, dtype, value):
        self.name = name
        self.dtype = dtype
        self.value = value

    def __str__(self):
        return "{} {} = {};\n\n".format(self.dtype, self.name, self.value)


class HeaderArray(HeaderEntry):
    def __init__(self, name, dtype, data, locality="RT_LOCAL_DATA"):
        assert locality in ["RT_LOCAL_DATA", "RT_L2_DATA", "RT_CL_DATA"]
        self.name = name
        self.dtype = dtype
        self.data = data
        self.locality = locality

    def __str__(self):
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
        ret += "};\n\n"
        return ret
