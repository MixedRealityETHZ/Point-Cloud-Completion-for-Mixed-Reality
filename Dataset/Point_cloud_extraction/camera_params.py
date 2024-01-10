from typing import NamedTuple
import numpy as np

# INTRINSICS
# For each camera we have the calibrated focal length and center position  [in px]

class RS4_01(NamedTuple):
    fx: float = 619.598
    fy: float = 619.116
    cx: float = 325.345
    cy: float = 245.441
    depth_unit: float = 1.0

class RS4_02(NamedTuple):
    fx: float = 615.85
    fy: float = 615.477
    cx: float = 316.062
    cy: float = 247.156
    depth_unit: float = 1.0

class RS4_03(NamedTuple):
    fx: float = 619.475
    fy: float = 619.189
    cx: float = 313.715
    cy: float = 223.921
    depth_unit: float = 1.0

class RS4_04(NamedTuple):
    fx: float = 615.665
    fy: float = 615.09
    cx: float = 306.514
    cy: float = 240.344
    depth_unit: float = 1.0


CameraIntrinsics = {
    '840412062035': RS4_01(),
    '840412062037': RS4_02(),
    '840412062038': RS4_03(),
    '840412062076': RS4_04(),
}
