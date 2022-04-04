#
# Copyright (C) 2018 Pico Technology Ltd. See LICENSE file for terms.
#
# ps6000 RAPID BLOCK MODE EXAMPLE
# This example opens a 6000 driver device, sets up one channel and a trigger then collects 10 block of data in rapid succession.
# This data is then plotted as mV against time in ns.

#import sys
import os
import time
import ctypes
from picosdk.ps6000 import ps6000 as ps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc
import psModule as pm

############# Constant values
chRange=[2,2,2,2,2,6] # ranges for each channel
# Voltage Ranges
# 2 = 50 mV
# 3 = 100 mV 
# 4 = 200 mV
# 5 = 500 mV
# 6 = 1 V
# 7 = 2 V
# 8 = 5 V 
# 9 = 10 V 
# 10 = 20 V
setCh=['setChA','setChB','setChC','setChD','EXT','AUX']
maxADC = ctypes.c_int16(32512) #should be same for both 3000 and 6000?
microsecond=1e-6
polarity=-1

TimeOutFlag=False
homeDir=os.environ["HOME"]

# Setting the number of sample to be collected
#preTriggerSamples  = 256
#postTriggerSamples = 256
#preTriggerSamples  = 250
#postTriggerSamples = 150
##### Test at Synchrotron
preTriggerSamples  = 750
postTriggerSamples = 3000
##### for LED measurement 2021.09.30
#preTriggerSamples  = 5
#postTriggerSamples = 395
maxSamples = preTriggerSamples + postTriggerSamples
print('max samples',maxSamples)
timebase=2 # 1.25GSPS
windowSize=70
startTime=preTriggerSamples
stopTime=startTime+windowSize
autotriggerCounter = 0
#autoTriggerMilliseconds = 7000
autoTriggerMilliseconds=0 #set to 0 for low rate cosmic ray scint-fibre test 24/02/21
plotEachFig=True
triggerChan=[4,5]

nev     =100
thr_mV  =10
runMode =0
nperplot=1
genPulseV=1000    # in micro-volts
genPulseRate=100  # in Hz

daqStartTime=0
daqEndTime  =0

# Create chandle and status ready for use
status = []
chandle = ctypes.c_int16()
connected=[]
init=False

timeIntervalns = ctypes.c_float()
tInts = []
# Creates converted types maxSamples
cmaxSamples = ctypes.c_int32(maxSamples)
fname=''
ofile={}
dataToSave={}
read_ch_en=[True,True,True,True]
trig_ch_en=[False,False,True,True]

#### Number of points for moving average
numAve=5


#Added variables
chRanges=[]
DevList=[]
nDevices=0
setDevList=[]
status_list=[]

def set_status(Array,DevList):
    status = []
    for i in range(len(DevList)):
        tstatus=[False,False,False,False]
        print("Device: ",DevList[i])
        if len(Array[i])==4:
            tstatus=[bool(int(Array[i][0])),bool(int(Array[i][1])),
                     bool(int(Array[i][2])),bool(int(Array[i][3]))]
            print(tstatus)
            status.append(tstatus)
    return status

def set_pulseParam(var0,var1):
    global genPulseV
    global genPulseRate
    genPulseV=var0
    genPulseRate=var1
    print('Pulse Voltage=',genPulseV*1e-3,' [mV], Pulse Rate=',genPulseRate*1e-6,' [MHz]')

def set_params(var0,var1,var2,var3,var4,var5,var6,var7):
    global nev
    global thr_mV
    global runMode
    global nperplot
    global dname
    global read_ch_en
    global trig_ch_en
    global chRanges
    global fname
    global DevList
    global setDevList
    global nDevices
    global status
    global connected
    global tInts
    #Input list of devices
    DevList = var7
    nDevices = len(var7)
    #Creat SetDevList
    for i in range(nDevices):
        CurrentList =  [c + DevList[i] for c in setCh]
        setDevList.append(CurrentList)
        status.append({})
        connected.append(False)
    
    tInts = np.ones(nDevices) #initiate time interval array
    print("Set Channel Status List: ",setDevList)  
    
   
    #Number of events
    nev     = var0
  
    #threshold voltage (same for each device)
    thr_mV  = var1
    
    #Run mode for each device
    runMode = var2
    
    #Folder Name for each device
    fname=var3
   
    #Check lengths
    if(len(var4)!=nDevices): print("Number of read channel settings != nDevices")
    if(len(var5)!=nDevices): print("Number of trigger channel settings != nDevices")
    if(len(var6)!=nDevices): print("Number of voltage range settings != nDevices")
 
    #Set status for each device
    print('Read ch status')
    read_ch_en=set_status(var4,var7)
    print('Trig ch status')
    trig_ch_en=set_status(var5,var7)
 
    for i in range(nDevices):
        print("Device: ",DevList[i])
        if len(var6[i])!=4:
            print('Wrong argument assigned for var6 in device',DevList[i])
            exit()
        tchRange = np.ones(6)*6
        tVr = var6[i]
        print(tVr)
        for ch in range(4): tchRange[ch]=tVr[ch]
        chRanges.append(tchRange.astype(int))
        print("ChRange[i]")
        print(tchRange)
    print('Number of events to be collected: ',nev)

def sig_gen():
    global status
    global chandle
    # Output a square wave with peak-to-peak voltage of 2 V and frequency of 10 kHz
    # handle = chandle
    # offsetVoltage = 1000000
    # pkToPk = 2000000
    # waveType = ctypes.c_int16(1) = PS6000_SQUARE
    # startFrequency = 1 MHz
    # stopFrequency = 1 MHz
    # increment = 0
    # dwellTime = 1
    # sweepType = ctypes.c_int16(1) = PS6000_UP
    # operation = 0
    # shots = 1
    # sweeps = 0
    # triggerType = ctypes.c_int16(0) = PS6000_SIGGEN_RISING
    # triggerSource = ctypes.c_int16(0) = PS6000_SIGGEN_NONE
    # extInThreshold = 1
    wavetype = ctypes.c_int16(1)
    sweepType = ctypes.c_int32(0)
    triggertype = ctypes.c_int32(0)
    #triggerSource = ctypes.c_int32(0)
    triggerSource = ctypes.c_int32(2) ## AUX
    #triggerSource = ps.PS6000_CHANNEL["PS6000_TRIGGER_AUX"]
    #status["SetSigGenBuiltIn"] = ps.ps6000SetSigGenBuiltIn(chandle, int(genPulseV/2), genPulseV, wavetype,
    status["SetSigGenBuiltIn"] = ps.ps6000SetSigGenBuiltIn(chandle, 0, genPulseV, wavetype,
                                                           genPulseRate, genPulseRate, 0, 1, sweepType, 0, 1, 0, triggertype, triggerSource, 1)
    time.sleep(1)
    print('BuiltIn Sig Gen is activated')
    assert_pico_ok(status["SetSigGenBuiltIn"])

def open_scope(nDev):
    #nDev = device no.
    global status
    global chandle
    global connected
    # Opens the device/s
    if (connected[nDev]==False):
        print("008")
        DevOpen = "openunit"+DevList[nDev]
        status[nDev][DevOpen] = pm.OpenUnit(chandle, DevList[nDev])
        print("009")
        assert_pico_ok(status[nDev][DevOpen])
        print("010")
        connected[nDev]=True
    
    # Displays the serial number and handle
    print(chandle.value)

def channel_init(channel,coupling,nDev):
    global status
    global chandle
    print('Init ch %s in device %s' %(channel,DevList[nDev]))
    # Set up channel A
    # handle = chandle
    # channel = ps6000_CHANNEL_A = 0
    # enabled = 1
    # coupling type = ps6000_DC_50R = 2
    # range = ps6000_50MV = 3
    # analogue offset = 0 V
    print("Coupling = ",coupling)
    Set=setDevList[nDev][channel]
    ch_range=chRanges[nDev][channel]
    Enable = 1
    Offset = 0
    Bandwidth = 0
    print("Ch Range = ",ch_range)
    status[nDev][Set] = pm.SetChannel(chandle, channel, Enable, coupling, ch_range, Offset, Bandwidth,DevList[nDev])
    print(status)
    print(Set,' ',ch_range,' ',coupling)
    assert_pico_ok(status[nDev][Set])
    return True

def set_advancedTrigger(value,chan_en,useAUX):
    # Set up window pulse width trigger on specified channel
    global status
    global chandle
    states=[ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
            ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"]]
    dirs=[ps.PS6000_THRESHOLD_DIRECTION["PS6000_NONE"],ps.PS6000_THRESHOLD_DIRECTION["PS6000_NONE"],
          ps.PS6000_THRESHOLD_DIRECTION["PS6000_NONE"],ps.PS6000_THRESHOLD_DIRECTION["PS6000_NONE"]]
    conds=[ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
           ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
           ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"]] 
    nUseCh=0
    for ch in range(4):
        if(chan_en[ch]==True):
            print('Ch ',ch,' are trigger channel')
            states[ch] = ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_TRUE"]
            conds[ch]  = ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_TRUE"]
            if polarity>0:
                dirs[ch]   = ps.PS6000_THRESHOLD_DIRECTION["PS6000_ABOVE"]
            else:
                dirs[ch]   = ps.PS6000_THRESHOLD_DIRECTION["PS6000_BELOW"]
            nUseCh = nUseCh+1
    print("TEMP: ",states)
    triggerConditions = ps.PS6000_TRIGGER_CONDITIONS(states[0],states[1],states[2],states[3],
                                                     ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
                                                     ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
                                                     ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_TRUE"]) # Use PWQ
    nTriggerConditions = 1
    status["setTriggerChannelConditions"] = ps.ps6000SetTriggerChannelConditions(chandle, ctypes.byref(triggerConditions), nTriggerConditions)
    assert_pico_ok(status["setTriggerChannelConditions"])
    ### use only INSIDE option for now..
    status["setTriggerChannelDirections"] = ps.ps6000SetTriggerChannelDirections(chandle,dirs[0],dirs[1],dirs[2],dirs[3], 
                                                                                 ps.PS6000_THRESHOLD_DIRECTION["PS6000_NONE"],
                                                                                 ps.PS6000_THRESHOLD_DIRECTION["PS6000_NONE"])
    assert_pico_ok(status["setTriggerChannelDirections"])
    
    # Sets up single trigger
    # Handle = Chandle
    # Enable = 1
    # Source = ps6000_channel_A = 0
    # Threshold = 800 ADC counts
    # Direction = ps6000_Rising
    # Delay = 0
    # autoTrigger_ms = 1000
    nChannelProperties = 0
    auxOutputEnable = 0
    ### Make an empty array of TRIGGER_CHANNEL_PROPERTIES with a length of "nUseCh"
    channelProperties=(ps.PS6000_TRIGGER_CHANNEL_PROPERTIES *nUseCh)()
    hysteresis = 0
    for ch in range(4):
        if(chan_en[ch]==False):continue
        print('### ',ch)
        ch_range=chRange[ch]
        threshold = mV2adc(value, ch_range, maxADC)
        if ch_range>8:
            threshold = mV2adc(0, ch_range, maxADC)
        if (ch_range==2):
            maxthreshold = mV2adc(50, ch_range, maxADC)
        elif (ch_range==3):
            maxthreshold = mV2adc(100, ch_range, maxADC)
        elif (ch_range==4):
            maxthreshold = mV2adc(200, ch_range, maxADC)
        elif (ch_range==5):
            maxthreshold = mV2adc(500,ch_range,maxADC)
        elif (ch_range==6):
            maxthreshold = mV2adc(1000,ch_range,maxADC)
        elif (ch_range==7):
            maxthreshold = mV2adc(2000, ch_range, maxADC)
        print('threshold=',value,', (', threshold,' in COUNT)')
        thre0 = min(threshold,maxthreshold)
        thre1 = max(threshold,maxthreshold)
        print("Thre0/Thre1 : ",thre0,"/",thre1)
        mode = ps.PS6000_THRESHOLD_MODE["PS6000_LEVEL"]
        channelProperties[nChannelProperties] = ps.PS6000_TRIGGER_CHANNEL_PROPERTIES(polarity*thre0,
                                                                 hysteresis,
                                                                 polarity*thre1,
                                                                 hysteresis,
                                                                 ch, mode)
        nChannelProperties+=1
    status["setTriggerChannelProperties"] = ps.ps6000SetTriggerChannelProperties(chandle, ctypes.byref(channelProperties),
                                                                                 nUseCh, auxOutputEnable, autoTriggerMilliseconds)
    assert_pico_ok(status["setTriggerChannelProperties"])

    ##### Following pulse width qualifier seems not properly working now..
    pwqConditions = ps.PS6000_PWQ_CONDITIONS(conds[0],conds[1],conds[2],conds[3],conds[4],conds[5])
    nPwqConditions = 1
    direction = 0
    #direction = ps.PS6000_THRESHOLD_DIRECTION["PS6000_RISING_OR_FALLING"]
    if polarity>0:
        #direction = ps.PS6000_THRESHOLD_DIRECTION["PS6000_RISING"]
        direction = ps.PS6000_THRESHOLD_DIRECTION["PS6000_ABOVE"]
    else:
        #direction = ps.PS6000_THRESHOLD_DIRECTION["PS6000_FALLING"]
        direction = ps.PS6000_THRESHOLD_DIRECTION["PS6000_BELOW"]
    #upper = 30 # x(time interval) (0.8ns now)
    #lower = upper
    upper = 20 # x(time interval) (0.8ns now)
    lower = 5
    ptype = ps.PS6000_PULSE_WIDTH_TYPE["PS6000_PW_TYPE_GREATER_THAN"]
    #ptype = ps.PS6000_PULSE_WIDTH_TYPE["PS6000_PW_TYPE_IN_RANGE"]
    #ptype = ps.PS6000_PULSE_WIDTH_TYPE["PS6000_PW_TYPE_NONE"]
    status["setPulseWidthQualifier"] = ps.ps6000SetPulseWidthQualifier(chandle, ctypes.byref(pwqConditions), nPwqConditions, direction, lower, upper, ptype)
    assert_pico_ok(status["setPulseWidthQualifier"])

###### Simple threshold trigger
def set_simpleTrigger(value,channel,rise,nDev):
    # value= threshold in mV,channel=source channel,dir=direction, nDev = device no.
    global status
    global chandle
    direction = pm.SetThresholdDirection(rise,DevList[nDev])
    print("### ", channel)
    print(value)
    print(type(value))
    threshold  = mV2adc(value, chRanges[nDev][channel], maxADC)
    print("HEEEEEEERE")
    print(direction)
    print(type(direction))
    print(chRanges[nDev][channel])
    print(type(chRanges[nDev][channel]))
    Set='trigger'+DevList[nDev]
    print('threshold=',value,', (', threshold,' in COUNT)')
    Enable = 1
    Delay = 0
    AutoTrig = autoTriggerMilliseconds
    print(Set)
    status[nDev][Set] = pm.SetSimpleTrigger(chandle, Enable, channel, threshold, direction, Delay, AutoTrig,DevList[nDev])
    print("Made it here 4")
    assert_pico_ok(status[nDev][Set])

def set_timebase(base,nDev):
    # Gets timebase innfomation
    # Handle = chandle
    # Timebase = 2 = timebase
    # Nosample = maxSamples
    # TimeIntervalNanoseconds = ctypes.byref(timeIntervalns)
    # MaxSamples = ctypes.byref(returnedMaxSamples)
    # Segement index = 0
    global timebase
    global timeIntervalns
    global tInts
    timebase = base
    returnedMaxSamples = ctypes.c_int16()
    SetGetTimebase="GetTimebase"+DevList[nDev]
    Oversample=1
    SegInd=0
    status[nDev][SetGetTimebase] = pm.GetTimebase2(chandle, timebase, maxSamples, ctypes.byref(timeIntervalns),Oversample, ctypes.byref(returnedMaxSamples), SegInd,DevList[nDev])
    tInts[nDev]=timeIntervalns.value
    assert_pico_ok(status[nDev][SetGetTimebase])

def get_single_event(nDev):
    global status
    global cmaxSamples
    global autotriggerCounter
    # Creates a overlow location for data
    overflow = ctypes.c_int16()

    # Handle = Chandle
    # nSegments = 1
    # nMaxSamples = ctypes.byref(cmaxSamples)
    SetMem = "MemorySegments"+DevList[nDev]
    nSeg = 1
    status[nDev][SetMem] = pm.MemorySegments(chandle, nSeg, ctypes.byref(cmaxSamples),DevList[nDev])
    assert_pico_ok(status[nDev][SetMem])
    
    # sets number of captures
    SetCapture="SetNoOfCaptures"+DevList[nDev]
    nCap = 1
    status[nDev][SetCapture] = pm.SetCaptures(chandle, nCap, DevList[nDev])
    assert_pico_ok(status[nDev][SetCapture])
    
    # Starts the block capture
    # Handle = chandle
    # Number of prTriggerSamples
    # Number of postTriggerSamples
    # Timebase = 2 = 4ns (see Programmer's guide for more information on timebases)
    # timebase = 2 is 0.8 ns for 6000 and 4 ns for 3000
    # time indisposed ms = None (This is not needed within the example)
    # segment index = 0
    # LpRead = None
    # pParameter = None
    SetBlock = "runBlock"+DevList[nDev]
    SegInd = 0
    OverSample = 0
    TimeIndis = None
    LpReady = None
    pParam = None
    status[nDev][SetBlock] = pm.SetRunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, OverSample, TimeIndis, SegInd, LpReady, pParam,DevList[nDev])
    assert_pico_ok(status[nDev][SetBlock])
    
    # Checks data collection to finish the capture
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    StartTime = time.time()
    SetIsReady = "isReady"+DevList[nDev]
    while ready.value == check.value:
        status[nDev][SetIsReady] = pm.IsReady(chandle, ctypes.byref(ready),DevList[nDev])
    EndTime = time.time()
    ElapsedTime = EndTime-StartTime
    #print("Elapsed time = ", ElapsedTime)
    if(autoTriggerMilliseconds>0):
        if(ElapsedTime>=autoTriggerMilliseconds/1000): autotriggerCounter+=1
    # Create buffers ready for assigning pointers for data collection
    bufferMax=[[],[],[],[]]
    bufferMin=[[],[],[],[]]
    status_str=["setDataBuffersA"+DevList[nDev],"setDataBuffersB"+DevList[nDev],
                "setDataBuffersC"+DevList[nDev],"setDataBuffersD"+DevList[nDev]]
    DownSampRatioMode = 0
    for ch in range(4):
        if read_ch_en[nDev][ch]==True:
            bufferMax[ch] = (ctypes.c_int16 * maxSamples)()
            bufferMin[ch] = (ctypes.c_int16 * maxSamples)() # used for downsampling which isn't in the scope of this example
            # Setting the data buffer location for data collection from channel [ch]
            status[nDev][status_str[ch]] = pm.SetDataBuffers(chandle, ch, ctypes.byref(bufferMax[ch]), 
                                                             ctypes.byref(bufferMin[ch]), maxSamples, 
                                                             SegInd,DownSampRatioMode,DevList[nDev])
            assert_pico_ok(status[nDev][status_str[ch]])
    
    # Handle = chandle
    # noOfSamples = ctypes.byref(cmaxSamples)
    # start index = 0
    # ToSegmentIndex = 0
    # DownSampleRatio = 1
    # DownSampleRatioMode = 0
    # Overflow = ctypes.byref(overflow)
    SetGetValues = "getValues"+DevList[nDev]
    StartIndex = 0
    DownRatio = 1
    status[nDev][SetGetValues] = pm.GetValues(chandle, StartIndex, ctypes.byref(cmaxSamples), DownRatio,
                                              DownSampRatioMode, SegInd, ctypes.byref(overflow),
                                              DevList[nDev])
    assert_pico_ok(status[nDev][SetGetValues])
    
    #### Below is only option for bulk readout
    ## Handle = chandle
    ## Times = Times = (ctypes.c_int16*10)() = ctypes.byref(Times)
    ## Timeunits = TimeUnits = ctypes.c_char() = ctypes.byref(TimeUnits)
    ## Fromsegmentindex = 0
    ## Tosegementindex = 0
    #Times = (ctypes.c_int16*1)()
    #TimeUnits = ctypes.c_char()
    #status["getValuesTriggerTimeOffsetBulk"] = ps.ps6000GetValuesTriggerTimeOffsetBulk64(chandle, ctypes.byref(Times), ctypes.byref(TimeUnits), 0, 0)
    #assert_pico_ok(status["getValuesTriggerTimeOffsetBulk"])

    # Converts ADC from channel 'ch' to mV
    #adc2mVChMax=[[],[],[],[]]
    data=[[],[],[],[]]
    for ch in range(4):
        if read_ch_en[nDev][ch]==True:
            #adc2mVChMax[ch]=adc2mV(bufferMax[ch], chRange[ch], maxADC)
            data[ch]=bufferMax[ch]
        else:
            #adc2mVChMax[ch]=0
            data[ch]=0
    #return adc2mVChMax
    return data

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N])/ float(N)

def analyse_and_plot_data(data,figname,nDev):
    # Plots the data from channel A onto a graph
    #global dataToSave

    base=np.array([])
    wfrms=np.array([])
    vmax=np.array([])
    charge=np.array([])
    fig = plt.figure(figsize=(15,8))
    gs = gridspec.GridSpec(4,4)
    ax1 = plt.subplot(gs[0,:])
    ax2 = plt.subplot(gs[1,:])
    ax3 = plt.subplot(gs[2,:])
    ax4 = plt.subplot(gs[3,:])

    TimeInt = tInts[nDev]
    timeX = np.linspace(0, (cmaxSamples.value) * TimeInt, cmaxSamples.value)
    startTime2 = startTime + numAve
    stopTime2  = stopTime  + numAve
    chStatus=read_ch_en+trig_ch_en
    nEvents = len(data)
    dataToSave=["BEGINHEADER",chStatus,daqStartTime,daqEndTime,nEvents,maxSamples]
    waveforms={}
    #### Convert data from digits to mV and perform the simple analysis
    for i in range(nEvents):
        #print('###',i)
        for ch in range(4):
            if read_ch_en[nDev][ch]==False: continue
            adc2mVChMax=np.array(adc2mV(data[i][ch], chRanges[nDev][ch], maxADC))
            #avwf = polarity * running_mean(adc2mVChMax[:],numAve)
            avwf = polarity * adc2mVChMax ### Skip averaging to speed up the daq..
            baseline =avwf[:windowSize].mean()
            chargeTmp=(avwf[startTime:stopTime].sum()-baseline*windowSize)
            if ch==0:
                base  = np.append(base, baseline)                     # mean value of the baseline
                wfrms = np.append(wfrms, avwf[:windowSize].std())     # standard deviation
                vmax  = np.append(vmax, avwf[startTime2:stopTime2].max()-baseline) # maximum voltage within the time window
                charge= np.append(charge, chargeTmp*float(TimeInt))   # integrated charge
            if len(waveforms)==0:
                waveforms=np.transpose(adc2mVChMax)
            else:
                waveforms=np.vstack([waveforms,np.transpose(adc2mVChMax)])
            if (i%nperplot)!=0:continue
            if ch==0:
                ax1.plot(timeX, adc2mVChMax[:]-baseline)
            if ch==1:
                ax2.plot(timeX, adc2mVChMax[:]-baseline)
            if ch==2:
                ax3.plot(timeX, adc2mVChMax[:]-baseline)
            if ch==3:
                ax4.plot(timeX, adc2mVChMax[:]-baseline)


    dataToSave.append(waveforms)

    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_xlim(-1,(cmaxSamples.value) * TimeInt + 1)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.set_xlim(-1,(cmaxSamples.value) * TimeInt + 1)
    
    #Added 24/02
    ax3.set_xlabel('Time(ns)')
    ax3.set_ylabel('Voltage(mV)')
    ax3.set_xlim(-1,(cmaxSamples.value) * TimeInt + 1)
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Voltage (mV)')
    ax4.set_xlim(-1,(cmaxSamples.value) * TimeInt + 1)    

    
    fig.savefig(homeDir+'/Desktop/'+figname)
    plt.close(fig)
    return dataToSave

### Initialise channel A & B
def init_daq():
    global TimeOutFlag
    global init
    global polarity
    global trig_ch_en
    global chRanges
    global nDevices
    global DevList
    global path
    
    path = homeDir+'/work/data/'+fname
    TimeOutFlag=False
    #print("004")
    #couplings=[ps.PS6000_COUPLING["PS6000_DC_50R"],ps.PS6000_COUPLING["PS6000_DC_50R"],
    #           ps.PS6000_COUPLING["PS6000_DC_50R"],ps.PS6000_COUPLING["PS6000_DC_50R"]]
    #print("FIRST TEST MADE HERE?")
    couplings = []
    for i in range(nDevices):
        couplings.append(pm.SetCouplings(DevList[i]))

    #if runMode==0: trig_ch_en=[False,False,True,False] ### !!Temporary
    #if runMode==3:
    #    trig_ch_en=[False,True,False,False] ### !!Temporary
    #    print("005")
    #    couplings[1] = ps.PS6000_COUPLING["PS6000_DC_1M"]

    #print("006")
    if init==False:
        #print("007")
        for i in range(nDevices):
            open_scope(i)
            print("Opened Device Pico", DevList[i])
            read_current_dev = read_ch_en[i]
            trig_current_dev = trig_ch_en[i]
            couplings_current_dev = couplings[i]

            for ch in range(4):
                if read_current_dev[ch]==True or trig_current_dev[ch]==True:
                    channel_init(ch,couplings_current_dev[ch],i) #added which device no. being set
           
            #if runMode==3:
            #    #sig_gen()
            #    polarity=+1
            ### pedestal run
            #if  runMode==0 or runMode==3:
            #    trigCh=-1
            #    for ch in range(4):
            #        if trig_current_dev[ch]==True:
            #            trigCh=ch
            #            break 
            #    if trigCh<0 or trigCh>3:
            #        print("Illegal Trigger Channel: ",trigCh)
            #        exit()
            #    print('Trigger ch: ',trigCh)
            #    set_simpleTrigger(polarity*thr_mV,trigCh,True,i)
            #if runMode==1: # negative polarity for SiPM signals
            #    polarity = -1
            #    set_advancedTrigger(thr_mV,trig_ch_en,False,i)
            #if runMode==2: # positive polarity for other tests
               # polarity = +1
               # set_advancedTrigger(thr_mV,trig_ch_en,False,i)
            
            #Only tested external trigger functions so far...
            if runMode==4: # Use AUX line for the triggering
                #channel_init(5,couplings[0])
                polarity = +1
                print("Checking...")
                print(polarity*thr_mV)
                print(type(polarity*thr_mV))
                set_simpleTrigger(polarity*thr_mV,5,True,i)
            else: print("Not tested this mode for multiple devices yet...")
            init=True
        #print("Polarity = ",polarity)
            #Make a path for each device
            DevPath = path+'/pico'+DevList[i]
            #Check whether the specified path exists or not
            isExist = os.path.exists(DevPath)
            print("Path name: ")
            print(DevPath)
            if not isExist:
               #Create a new directory because it does not exist 
               os.makedirs(DevPath)
               print("The new directory is created!")


def getTimeOutFlag():
    return TimeOutFlag

def run_daq(sub,run,nDev):
    set_timebase(2,nDev) ## 1.25GSPS
    data=[]
    global ofile
    #global dataToSave #I dont think it should be global if running multiple at same time?
    global daqStartTime
    global daqEndTime
    global autotriggerCounter
    global TimeOutFlag
    global init
    global connected
    global tInts
    TimeInt = tInts[nDev]

    ##Make a path for each device
    #path = homeDir+'/work/data/'+fname+'/pico'+DevList[nDev]
    ## Check whether the specified path exists or not
    #isExist = os.path.exists(path)
    #print("Path name: ")
    #print(path)
    #if not isExist:
    #    # Create a new directory because it does not exist 
    #    os.makedirs(path)
    #    print("The new directory is created!")
    
    
    if(run!=0): fname_sub = path+'/data'+str(run)+'_'+str(sub)+'.npy'
    elif(run==0): fname_sub=path+'/data'+str(sub)+'.npy'
    
    ofile=open(fname_sub,"wb")
    #print('time interval = ',timeIntervalns.value)
    print('integration from ',startTime*TimeInt,' to ',stopTime*TimeInt,' [ns]')
    daqStartTime=time.time()
    for iev in range(nev):
        #adc2mVData=get_single_event()
        #data.append(adc2mVData)
        rawdata=get_single_event(nDev)
        data.append(rawdata)
        if(autotriggerCounter>2): 
            print("Timeout occurred!")
            TimeOutFlag=True
            autotriggerCounter=0
            init = False
            connected = False
            break
        time.sleep(100*microsecond) ### Trigger rate is limited to 10kHz here.
    daqEndTime=time.time()
    
    print('Trigger rate = ', float(nev)/(daqEndTime-daqStartTime), ' Hz'), 
    # Stops the scope
    # Handle = chandle
    SetStop = "stop"+DevList[nDev]
    status[nDev][SetStop] = pm.StopScope(chandle,DevList[nDev])
    assert_pico_ok(status[nDev][SetStop])
    #dataToSave={} #Init
    dataToSave = analyse_and_plot_data(data,'fig_pico'+DevList[nDev]+'.png',nDev)
    np.save(ofile,dataToSave,allow_pickle=True)
    ofile.close

def close(nDev):
    # Closes the unit
    # Handle = chandle
    SetClose="close"+DevList[nDev]
    status[nDev][SetClose] = pm.CloseScope(chandle,DevList[nDev])
    assert_pico_ok(status[nDev][SetClose])
    
    # Displays the staus returns
    print(status)

