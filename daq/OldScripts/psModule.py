import ctypes
from picosdk.ps3000a import ps3000a as ps3
from picosdk.ps6000 import ps6000 as ps6
from picosdk.functions import adc2mV, mV2adc,assert_pico_ok


def OpenUnit(chandle, device):
    if device=="3000":
        return ps3.ps3000aOpenUnit(ctypes.byref(chandle), None)
    elif device=="6000":
        return ps6.ps6000OpenUnit(ctypes.byref(chandle), None)
    else:
        print("ERROR: Invalid Device (OpenUnit)")
        return

def SetCouplings(device):
    if device=="3000":
       couplings = [ps3.PS3000A_COUPLING["PS3000A_DC"],ps3.PS3000A_COUPLING["PS3000A_DC"],
                    ps3.PS3000A_COUPLING["PS3000A_DC"],ps3.PS3000A_COUPLING["PS3000A_DC"]]
       return couplings
    
    elif device=="6000":
       couplings = [ps6.PS6000_COUPLING["PS6000_DC_50R"],ps6.PS6000_COUPLING["PS6000_DC_50R"],
                    ps6.PS6000_COUPLING["PS6000_DC_50R"],ps6.PS6000_COUPLING["PS6000_DC_50R"]]
       return couplings
    else:
       print("ERROR: Invalid Device (SetCouplings)")
       return

def SetChannel(chandle,channel,enable,coupling,ch_range,offset,bandwidth,device):
    if device == "3000":
       return ps3.ps3000aSetChannel(chandle,channel,enable,coupling,ch_range,offset)
    elif device == "6000":
       return ps6.ps6000SetChannel(chandle,channel,enable,coupling,ch_range,offset,bandwidth)
    else:
       print("ERROR: Invalid Device (SetChannel)")
       return

def SetThresholdDirection(rise,device):
    #rise (True or False) sets whether it is rising or falling edge for threshold
    if device == "3000":
        if rise == True:
            return ps3.PS3000A_THRESHOLD_DIRECTION["PS3000A_RISING"]
        else:
            return ps3.PS3000A_THRESHOLD_DIRECTION["PS3000A_FALLING"]
    
    elif device == "6000":
        if rise == True:
            return ps6.PS6000_THRESHOLD_DIRECTION["PS6000_RISING"]
        else: 
            return ps6.PS6000_THRESHOLD_DIRECTION["PS6000_FALLING"]
    
    else:
        print("ERROR: Invalid Device (SetThresholdDirection)") 
        return

def SetSimpleTrigger(chandle,enable,channel,threshold,direction,delay,autoTrig,device):
    if device =="3000":
        if channel==5: channel = 4 #3000 device uses external input instead of aux for our purposes
        return ps3.ps3000aSetSimpleTrigger(chandle,enable,channel,threshold,direction,delay,autoTrig)
    elif device == "6000":
        return ps6.ps6000SetSimpleTrigger(chandle,enable,channel,threshold,direction,delay,autoTrig)
    else:
        print("ERROR: Invalid Device (SetSimpleTrigger)")
        return

def StopScope(chandle,device):
    if device =='3000':
        return ps3.ps3000aStop(chandle)
    elif device =='6000':
        return ps6.ps6000Stop(chandle)
    else:
        print("ERROR: Invalid Device (StopScope)")
        return

def CloseScope(chandle,device):
    if device =='3000':
        return ps3.ps3000aCloseUnit(chandle)
    elif device =='6000':
        return ps6.ps6000CloseUnit(chandle)
    else:
        print("ERROR: Invalid Device (CloseScope)")
        return

def MemorySegments(chandle,nSegments,maxSamples,device):
    if device =='3000':
        return ps3.ps3000aMemorySegments(chandle,nSegments,maxSamples)
    elif device =='6000':
        return ps6.ps6000MemorySegments(chandle,nSegments,maxSamples)
    else:
        print("ERROR: Invalid Device (MemorySegments)")
        return

def SetCaptures(chandle,nCaptures,device):
    if device =='3000':
        return ps3.ps3000aSetNoOfCaptures(chandle,nCaptures)
    elif device =='6000':
        return ps6.ps6000SetNoOfCaptures(chandle,nCaptures)
    else:
        print("ERROR: Invalid Device (SetCaptures)")
        return

 
def SetRunBlock(chandle,PreTrigSamples,PostTrigSamples,TimeBase,Oversample,
                TimeIndisposed,SegmentIndex,LPReady,PParameter,device):
    if device =='3000':
        return ps3.ps3000aRunBlock(chandle,PreTrigSamples,PostTrigSamples,TimeBase,Oversample,
                                   TimeIndisposed,SegmentIndex,LPReady,PParameter)
    elif device =='6000':
        return ps6.ps6000RunBlock(chandle,PreTrigSamples,PostTrigSamples,TimeBase,Oversample,
                                  TimeIndisposed,SegmentIndex,LPReady,PParameter)
    else:
        print("ERROR: Invalid Device (SetRunBlock)")
        return

def IsReady(chandle,ready,device):
    if device =='3000':
        return ps3.ps3000aIsReady(chandle,ready)
    elif device =='6000':
        return ps6.ps6000IsReady(chandle,ready)
    else:
        print("ERROR: Invalid Device (IsReady)")
        return


def SetDataBuffers(chandle,channel,bufMax,bufMin,bufLen,SegmentIndex,downRatioMode,device):
    if device =='3000':
        return ps3.ps3000aSetDataBuffers(chandle,channel,bufMax,bufMin,bufLen,SegmentIndex,downRatioMode)
    elif device =='6000':
        return ps6.ps6000SetDataBuffers(chandle,channel,bufMax,bufMin,bufLen,SegmentIndex,downRatioMode)
    else:
        print("ERROR: Invalid Device (SetDataBuffers)")
        return



def GetValues(chandle,startI,nSamples,downRatio,downRatioMode,SegmentIndex,Overflow,device):
    if device =='3000':
        return ps3.ps3000aGetValues(chandle,startI,nSamples,downRatio,downRatioMode,SegmentIndex,
                                    Overflow)
    elif device =='6000':
        return ps6.ps6000GetValues(chandle,startI,nSamples,downRatio,downRatioMode,SegmentIndex,
                                   Overflow)
    else:
        print("ERROR: Invalid Device (GetValues)")
        return

def GetTimebase2(chandle,Timebase,nSamples,TimeInt,Oversample,maxSamples,SegmentIndex,device):
    if device =='3000':
        return ps3.ps3000aGetTimebase2(chandle,Timebase,nSamples,TimeInt,Oversample,maxSamples,
                                       SegmentIndex)
    elif device =='6000':
        return ps6.ps6000GetTimebase2(chandle,Timebase,nSamples,TimeInt,Oversample,maxSamples,
                                      SegmentIndex)
    else:
        print("ERROR: Invalid Device (GetValues)")
        return




