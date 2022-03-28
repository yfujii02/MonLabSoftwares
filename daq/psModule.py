import ctypes
from picosdk.ps3000 import ps3000 as ps3
from picosdk.ps6000 import ps6000 as ps6


def OpenUnit(chandle, device):
    if device=="3000":
        return ps3.ps3000OpenUnit(ctypes.byref(chandle), None)
    elif device=="6000":
        return ps6.ps6000OpenUnit(ctypes.byref(chandle), None)
    else:
        return "Error"
