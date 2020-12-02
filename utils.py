from enum import Enum
import re

class ParameterType(Enum):
    CNN_WEIGHTS = 1
    CNN_BIAS = 2
    FC_WEIGHTS = 3
    FC_BIAS = 4
    BN_WEIGHT = 5
    BN_BIAS = 6
    DOWNSAMPLE_WEIGHTS = 7
    DOWNSAMPLE_BIAS = 8
    DOWNSAMPLE_BN_W = 9
    DOWNSAMPLE_BN_B = 10

def int_from_str(str):
    return list(map(int, re.findall(r'\d+', str)))