import os
import time
import ctypes
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import psModule as pm
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc

#Global variables
maxADC = ctypes.c_int16(32512) #define max ADC value
homeDir = os.environ["HOME"] #home directory path
connected = False #status flag for whether scope has been opened or not
status = {} #store various pico status conditions
chandle = ctypes.c_int16()
initialised = False #check whether the picoscope has been initialised
TimeInt = ctypes.c_float() #create time interval variable (ns)
ofile = {} #create global data file open variable
dataSave = {} #data variable to store waveform data that will be saved
StatChan = ['SetChA_','SetChB_','SetChC_','SetChD_','EXT_','AUX_'] #strings for channel init status 
TrigPolarity = 1 #initiate variable for simple trigger polarity
daqStart = 0 #daq start time for each sub run
daqEnd = 0 #daq end time variable for each sub run
BaseWindow = 100 #starting window to extract a mean baseline

#Other Settings for pico functions (generally not changed)
#Channel Init
EnableCh = 1
OffsetCh = 0
BandwidthCh  = 0
#Set Simple Trigger
EnableTrig = 1
DelayTrig = 1
AutoTrig = 0 #auto trigger in milliseconds - 0 disables
#Set Timebase
OversampleTB = 1
SegIndTB = 0
#Set Memory Segment
nSegMS = 1
#Set Number of Captures
nCapNC = 1
#Set RunBlock
SegIndRB = 0
OversampleRB=0
TimeIndisposedRB = None
LPReadyRB = None
PParamRB = None
#Set DataBuffers
DownRatioModeDB = 0
SegIndDB=0
#Get Values
DownRatioGV=1
DownRatioModeGV=0
StartIndexGV=0
SegIndGV=0

#Initialise device and daq parameters
Device = "3000" #picoscope type - used to call correct pico library functions in psModule.py
DeviceNumber='1' #device number (for future case if have same types of pico)
Nevents = 100 #number of events to be recorded in a sub run
TrigThresh = [0,0,0,0,0,0] #trigger threshold used (if more than one ch is acting as trigger)
DAQMode = 1 #DAQ run mode (1: Simple single ch trigger)
fName = "temp" #folder name to store data in
fPath = "/home/comet/work/data/temp" #path to where data folder is
readCh_en = [0,0,0,0,0,0] #channels for data to be read
trigCh_en = [0,0,0,0,0,0] #trigger channels to be used (if more than one ch is acting as trigger)
SimpleTrigCh = 5 #trigger channel to be used (if only one ch is acting as trigger)
chRange = [1,1,1,1,1,1] #voltage range used for each channel
Polarities = [1,1,1,1,1,1] #polarity of the signal in each ch (useful for defining trigger)
Timebase = 2 #timebase to be used for this device (t = 2^n/S -> n =timebase, S = sampling rate)
             #For reference, our pico6000: S = 5e9, and pico3000: S = 1e9
             #Also this only applies for n = [0,2] for pico3000, and n = [0,4] for pico6000
             #Using n = 2, pico6000 has t = 0.8 ns time intervals, and pico3000 has 4 ns
preSamps = 0 #No. samples collected before trigger
postSamps = 0 #No. samples collected after trigger
maxSamples = 0 #Total no. samples collected per trigger (at least as with current settings, can be downsized)
cmaxSamples = ctypes.c_int32(0) #Total no. samples collected as a ctype variable


def ConvertStatus(string):
    ChStat = []
    for i in range(len(string)):
        ChStat.append(bool(int(string[i])))
    return ChStat

def set_params(DevInfo,DAQInfo,ChInfo,Stat):
    global status
    global chandle
    status = Stat[0]
    chandle = Stat[1]
    #Read in and define all of the device + daq parameters
    global Device 
    Device = DevInfo["typeDev"]
    global DeviceNumber
    DeviceNumber = DevInfo["numDev"]
    global Nevents 
    Nevents = DAQInfo["Nevents"]
    global TrigThresh 
    TrigThresh = ChInfo["Tthresh"]
    global DAQMode 
    DAQMode = DAQInfo["Mode"]
    global fName
    fName = DAQInfo["fName"]
    global fPath
    fPath = DAQInfo["fPath"]+DAQInfo["fName"]
    global readCh_en 
    readCh_en = ConvertStatus(ChInfo["Rstatus"])
    global trigCh_en 
    trigCh_en = ConvertStatus(ChInfo["Tstatus"])
    global SimpleTrigCh
    SimpleTrigCh = ChInfo["Tsimple"] 
    global chRange 
    chRange = ChInfo["Vrange"]
    global TimeBase
    TimeBase = DAQInfo["Tbase"]
    global preSamps
    preSamps = DAQInfo["preSamples"]
    global postSamps
    postSamps = DAQInfo["postSamples"]
    global maxSamples
    maxSamples = preSamps+postSamps
    global cmaxSamples
    cmaxSamples=ctypes.c_int32(maxSamples)
    global Polarities 
    Polarities = ChInfo["Polarity"]
    global BaseWindow
    BaseWindow = int(preSamps/5)
    
def open_scope():
    global status
    global chandle
    global connected
    StatOpen = "OpenUnit_"+Device
    if(connected==False):
        status[StatOpen] = pm.OpenUnit(chandle,Device)
        assert_pico_ok(status[StatOpen])
        connected = True
        print("Opening Pico_"+Device+" #"+DeviceNumber) #prints serial number and handle

def channel_init(channel,coupling):
    global status
    global chandle
    print("Init Ch ",channel," with coupling ",coupling)
    SCh = StatChan[channel]+Device
    RCh = chRange[channel]
    status[SCh] = pm.SetChannel(chandle,channel,EnableCh,coupling,RCh,
                                OffsetCh,BandwidthCh,Device)
    assert_pico_ok(status[SCh])
    return True

def SetTimeBase():
    global status
    global chandle
    global TimeInt
    retMaxSamples = ctypes.c_int16()
    STB = "TimeBase_"+Device
    status[STB] = pm.GetTimebase2(chandle,TimeBase,maxSamples,ctypes.byref(TimeInt),
                                  OversampleTB,ctypes.byref(retMaxSamples),SegIndTB,Device)  
    assert_pico_ok(status[STB])

def SetSimpleTrigger():
    global status
    global chandle
    
    #Channel, polarity and threshold of trigger determined from settings file 
    direction = pm.SetThresholdDirection(bool(TrigPolarity),Device) #direction of trigger
    #Threshold in mV = TrigPolarity * TrigThresh [SimpleTrigCh], converted to ADC count
    threshADC = mV2adc(TrigPolarity*TrigThresh[SimpleTrigCh],chRange[SimpleTrigCh],maxADC)
    
    STrig = "Trigger_"+Device+"_Ch_"+str(SimpleTrigCh)
    status[STrig] = pm.SetSimpleTrigger(chandle,EnableTrig,SimpleTrigCh,threshADC,direction,DelayTrig,AutoTrig,Device)
    assert_pico_ok(status[STrig])

#def SetAdvancedTrigger():
#    global status
#    global chandle

#    NCh = len(trigCh_en)
#    if(NCh!=6): print("ERROR: Not enough channels listed in settings!")
#    ChConds = []
#    ChDirs = []
#    nTrigCh=0
    
#    for ch in range(NCh):
#        ChDirs.append(pm.SetThresholdDirection(bool(Polarities[ch]),Device))
#        if(trigCh_en[ch]==True):
#            print('Ch ',ch,' is being set as a trigger channel')
#            ChConds.append(pm.UpdateTriggerState(1,Device))
#            nTrigCh += 1
#        else:
#            ChConds.append(pm.UpdateTriggerState(0,Device))

#    PWQ = pm.UpdateTriggerState(1,Device)        
#    nTrigConds = 1 
#    STrigCh = "SetTriggerChConditions_"+Device
#    status[STrigCh] = pm.SetTriggerConditions(chandle,ChConds,PWQ,nTrigConds,Device)
#    assert_pico_ok(status[STrigCh])

#    STrigDir = "SetTriggerChDirections_"+Device
#    status[STrigDir] = pm.SetTriggerConditions(chandle,ChDirs,Device)
#    assert_pico_ok(status[STrigDir])
# INCOMPLETE - FIX LATER!
    



def GetSingleEvent():
    global status
    global chandle
    global cmaxSamples
    
    Overflow = ctypes.c_int16() #create overflow location for data
    SMem = "MemSegments_"+Device
   
    status[SMem] = pm.MemorySegments(chandle,nSegMS,ctypes.byref(cmaxSamples),Device)
    assert_pico_ok(status[SMem])
    #Set number of captures
    SCap = "SetNoOfCaptures_"+Device
    status[SCap] = pm.SetCaptures(chandle,nCapNC,Device)
    assert_pico_ok(status[SCap]) 
   
    #Start block capture
    SBlock = "RunBlock_"+Device
    status[SBlock] = pm.SetRunBlock(chandle,preSamps,postSamps,TimeBase,OversampleRB,TimeIndisposedRB,
                                    SegIndRB,LPReadyRB,PParamRB,Device)
    #print("Not here yet?")
    assert_pico_ok(status[SBlock])
    
    #Checks data colletion to finish the capture
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
  
    # Wait until ready 
    #print("Ready")
    SReady = "IsReady_"+Device
    while ready.value == check.value: 
        status[SReady] = pm.IsReady(chandle,ctypes.byref(ready),Device)
    #print("Done")
    # Create buffers ready for assigning pointers for data collection
    bufferMax=[[],[],[],[]]
    bufferMin=[[],[],[],[]]
    status_str=["SetDataBuffersA_"+Device,"SetDataBuffersB_"+Device,
                "SetDataBuffersC_"+Device,"SetDataBuffersD_"+Device]
    
    for ch in range(4):
        if readCh_en[ch]==True:
            bufferMax[ch] = (ctypes.c_int16 * maxSamples)()
            bufferMin[ch] = (ctypes.c_int16 * maxSamples)() #can be used for downsampling
           
            # Setting the data buffer location for data collection from channel [ch]
            status[status_str[ch]] = pm.SetDataBuffers(chandle, ch, ctypes.byref(bufferMax[ch]),
                                                       ctypes.byref(bufferMin[ch]), maxSamples,SegIndDB,
                                                       DownRatioModeDB,Device)
            assert_pico_ok(status[status_str[ch]])
  
    #Get values from buffer
    SGetVals = "GetValues_"+ Device
    status[SGetVals] = pm.GetValues(chandle, StartIndexGV, ctypes.byref(cmaxSamples), DownRatioGV, 
                                             DownRatioModeGV, SegIndGV, ctypes.byref(Overflow),
                                             Device)
    assert_pico_ok(status[SGetVals])

    data=[[],[],[],[]]
    for ch in range(4):
        if readCh_en[ch]==True:
            data[ch]=bufferMax[ch]
        else:
            data[ch]=0 #don't care about ch data we don't want to read
     
    return data


def init_daq(DevInfo,DaqInfo,ChanInfo):
    global initialised
    global connected
    global status
    global chandle
   
    set_params(DevInfo,DaqInfo,ChanInfo,[status,chandle])
    #print(DevInfo)
    #print(DaqInfo)
    #print(ChanInfo)
    initialised = False
    connected = False
     
    couplings = pm.SetCouplings(Device)
    if initialised == False:
        open_scope()
        for ch in range(4):
            if readCh_en[ch]==True or trigCh_en[ch]==True: 
                channel_init(ch,couplings[ch])
        if DAQMode == 1:
#            print("Setting simple trigger")
            SetSimpleTrigger()
    
    initialised = True
    CheckDir = os.path.exists(fPath)
    if not CheckDir: os.makedirs(fPath)
    #print("Status at end of init")
    #print(status)
    StatRet = [status,chandle]
    status = {}
    chandle = ctypes.c_int16()
    return StatRet

def analyseData(data,figname):
    global dataSave
    tint = TimeInt.value
     
    base = np.array([])
    wfrms=np.array([])
    vmax=np.array([])
    charge=np.array([])
    fig = plt.figure(figsize=(15,8))
    gs = gridspec.GridSpec(4,4)
    ax1 = plt.subplot(gs[0,:])
    ax2 = plt.subplot(gs[1,:])
    ax3 = plt.subplot(gs[2,:])
    ax4 = plt.subplot(gs[3,:])

    timeX = np.linspace(0, (cmaxSamples.value) * tint, cmaxSamples.value)
   
    chStatus=readCh_en+trigCh_en
    nEv = len(data)
    dataSave=["BEGINHEADER",chStatus,daqStart,daqEnd,nEv,maxSamples]
    waveforms={}
  
    #### Convert data from digits to mV
    for i in range(nEv):
        for ch in range(4):
            if readCh_en[ch]==False: continue
            
            adc2mVChMax=np.array(adc2mV(data[i][ch], chRange[ch], maxADC))
            #avwf = Polarities[ch] * adc2mVChMax ### Skip averaging to speed up the daq..
            #baseline =avwf[:BaseWindow].mean()
            baseline=0
            if len(waveforms)==0:
                waveforms=np.transpose(adc2mVChMax)
            else:
                waveforms=np.vstack([waveforms,np.transpose(adc2mVChMax)])
            
            if ch==0:
                ax1.plot(timeX, adc2mVChMax[:]-baseline)
            if ch==1:
                ax2.plot(timeX, adc2mVChMax[:]-baseline)
            if ch==2:
                ax3.plot(timeX, adc2mVChMax[:]-baseline)
            if ch==3:
                ax4.plot(timeX, adc2mVChMax[:]-baseline)


    dataSave.append(waveforms)

    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_xlim(-1,(cmaxSamples.value) * tint + 1)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.set_xlim(-1,(cmaxSamples.value) * tint + 1)

   
    ax3.set_xlabel('Time(ns)')
    ax3.set_ylabel('Voltage(mV)')
    ax3.set_xlim(-1,(cmaxSamples.value) * tint + 1)
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Voltage (mV)')
    ax4.set_xlim(-1,(cmaxSamples.value) * tint + 1)


    fig.savefig(homeDir+'/Desktop/'+figname)
    plt.close(fig)
    

def CheckRunFlag():
    return FinishRunFlag

def run_daq(sub,Settings,Stat,RetStats,Ind):
    global ofile
    global dataSave
    global daqStart
    global daqEnd
    global initialised
    global connected
    global status
    global chandle

  
    #FinishRunFlag = False
    print(Settings[0])
    print(Settings[1])
    print(Settings[2])
    set_params(Settings[0],Settings[1],Settings[2],Stat)
    SetTimeBase()

    fName_file = fPath+'/data'+str(sub)+'.npy'
    ofile = open(fName_file,"wb")
    daqStart = time.time()
    data = []


    for iEv in range(Nevents):
        #print("Event ",iEv,"/",Nevents," in device:",Device)
        rawdata = GetSingleEvent()
        data.append(rawdata)
        time.sleep(100e-6)
     
    daqEnd = time.time()
   
    print('Trigger rate = ',float(Nevents)/(daqEnd-daqStart),' Hz')
    SStop = "Stop_"+Device
    status[SStop]=pm.StopScope(chandle,Device)
    assert_pico_ok(status[SStop])
    dataSave = {}
    print("Analysing and saving data from device ", Device)
    analyseData(data,"fig_pico_"+Device+".png")
    np.save(ofile,dataSave,allow_pickle=True)
    ofile.close
    print("Subrun complete in device ",Device)
    RetStats[Ind] = [status,chandle]
    RetStats[Ind+1] = True
    status = {}
    chandle = ctypes.c_int16()

def close(Settings,Stat):
    global status
    global chandle
    set_params(Settings[0],Settings[1],Settings[2],Stat)
    SClose = "Close_"+Device
    status[SClose] = pm.CloseScope(chandle,Device)
    assert_pico_ok(status[SClose])
    status = {}
    chandle = ctypes.c_int16()

