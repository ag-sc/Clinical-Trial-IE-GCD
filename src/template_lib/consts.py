import os
import sys

data_path = os.path.join(os.path.dirname(sys.modules["template_lib"].__file__), "..", "..", "data")
template_lib_path = os.path.dirname(sys.modules["template_lib"].__file__)