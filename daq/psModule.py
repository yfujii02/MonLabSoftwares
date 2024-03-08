import ctypes
from picosdk.ps3000a import ps3000a as ps3
from picosdk.ps6000 import ps6000 as ps6

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
       couplings = [ps6.PS6000_COUPLING["PS6000_DC_1M"],ps6.PS6000_COUPLING["PS6000_DC_1M"],
                    ps6.PS6000_COUPLING["PS6000_DC_1M"],ps6.PS6000_COUPLING["PS6000_DC_1M"]]
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

def SetThresholdDirection(polarity,rise,device):
    if device == "3000":
        if polarity == True:
            if rise == True: return ps3.PS3000A_THRESHOLD_DIRECTION["PS3000A_RISING"]
            else: return ps3.PS3000A_THRESHOLD_DIRECTION["PS3000A_ABOVE"]
        else: 
            if rise == True: return ps3.PS3000A_THRESHOLD_DIRECTION["PS3000A_FALLING"]
            else: return ps3.PS3000A_THRESHOLD_DIRECTION["PS3000A_BELOW"]
    
    elif device == "6000":
    
        if polarity == True:
            if rise == True: return ps6.PS6000_THRESHOLD_DIRECTION["PS6000_RISING"]
            else: return ps6.PS6000_THRESHOLD_DIRECTION["PS6000_ABOVE"]
        else: 
            if rise == True: return ps6.PS6000_THRESHOLD_DIRECTION["PS6000_FALLING"]
            else: return ps6.PS6000_THRESHOLD_DIRECTION["PS6000_BELOW"]
   
    else:
        print("ERROR: Invalid Device (SetThresholdDirection)") 
        return

def SetSimpleTrigger(chandle,enable,channel,threshold,direction,delay,autoTrig,device):
    if device =="3000":
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

def InitialChStates(device):
    if(device=='3000'):
        return [ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"]]
    elif(device=='6000'):
        return [ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
                ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
                ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
                ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
                ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
                ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"]]
    else:
        print("ERROR: Invalid Device (InitalChStates)")
        return

def UpdateTriggerState(state,device):
    #returns a specified trigger state: 0: DONT_CARE, 1: TRUE, 2: FALSE, 3: MAX
    if(device == '3000'):
        if(state==0):
            return ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"]
        elif(state==1):
            return ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_TRUE"]
        elif(state==2):
            return ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_FALSE"]
        elif(state==3):
            return ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_MAX"]
        else:
            print('Invalid state! Setting to DONT_CARE')
            return ps3.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"]
    elif(device == '6000'):
        if(state==0):
            return ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"]
        elif(state==1):
            return ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_TRUE"]
        elif(state==2):
            return ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_FALSE"]
        elif(state==3):
            return ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_MAX"]
        else:
            print('Invalid state! Setting to DONT_CARE')
            return ps6.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"]
    else:
        print('ERROR: Invalid Device (UpdateTriggerState)')
        return

def SetTriggerConditions(chandle,ch_states,pwq_state,ncond,device):
    st = ch_states+pwq_state
    if(device == '3000'):
        TrigCons = ps3.PS3000A_TRIGGER_CONDITIONS(st[0],st[1],st[2],st[3],st[4],st[5],st[6])
        return ps3.ps3000aSetTriggerChannelConditions(chandle,ctypes.byref(TrigCons),ncond)
    elif(device == '6000'):
        TrigCons = ps6.PS6000_TRIGGER_CONDITIONS(st[0],st[1],st[2],st[3],st[4],st[5],st[6])
        return ps6.ps6000SetTriggerChannelConditions(chandle,ctypes.byref(TrigCons),ncond)
    else:
        print('ERROR: Invalid Device (SetTriggerConditions)')
        return

def SetTriggerDirections(chandle,ch_dirs,device):
    d = ch_dirs
    if(device == '3000'):
        return ps3.ps3000aSetTriggerChannelDirections(chandle,d[0],d[1],d[2],d[3],d[4],d[5])
    elif(device == '6000'):
        return ps6.ps6000SetTriggerChannelDirections(chandle,d[0],d[1],d[2],d[3],d[4],d[5])
    else:
        print('ERROR: Invalid Device (SetTriggerDirections)')
        return

def SetThresholdMode(device):
    if(device=='3000'): return ps3.PS3000A_THRESHOLD_MODE["PS3000A_LEVEL"]
    elif(device=='6000'): return ps6.PS6000_THRESHOLD_MODE["PS6000_LEVEL"]
    else:
        print('ERROR: Invalid Device (SetThresholdMode)')
        return

def RetTrigProp(device):
    if(device=='3000'): return ps3.PS3000A_TRIGGER_CHANNEL_PROPERTIES
    elif(device=='6000'): return ps6.PS6000_TRIGGER_CHANNEL_PROPERTIES
    else:
        print('ERROR: Invalid Device (RetTrigProp)')
        return

def RetTrigChanProp(polarity,minT,maxT,hyst,channel,mode,device):
    if(device=='3000'): 
        return ps3.PS3000A_TRIGGER_CHANNEL_PROPERTIES(polarity*minT,hyst,polarity*maxT,
                                                      hyst,channel,mode) 
    elif(device=='6000'):
        return ps6.PS6000_TRIGGER_CHANNEL_PROPERTIES(polarity*minT,hyst,polarity*maxT,
                                                      hyst,channel,mode) 
    else:
        print('ERROR: Invalid Device (RetTrigChanProp)')
        return


def SetTrigChanProp(chandle,ch_props,nCh,auxEn,autoTrig,device):
    if(device=='3000'): 
        return ps3.ps3000aSetTriggerChannelProperties(chandle,ctypes.byref(ch_props),
                                                      nCh,auxEn,autoTrig) 
    elif(device=='6000'):
        return ps6.ps6000SetTriggerChannelProperties(chandle,ctypes.byref(ch_props),
                                                      nCh,auxEn,autoTrig) 
    else:
        print('ERROR: Invalid Device (SetTrigChanProp)')
        return


def SetPWQConds(ch_cons,device):
    c = ch_cons
    if(device == '3000'):
        return ps3.PS3000A_PWQ_CONDITIONS(c[0],c[1],c[2],c[3],c[4],c[5])
    elif(device == '6000'):
        return ps6.PS6000_PWQ_CONDITIONS(c[0],c[1],c[2],c[3],c[4],c[5])
    else:
        print('ERROR: Invalid Device (SetPWQConds)')
