#
# Copyright (C) 2018 Pico Technology Ltd. See LICENSE file for terms.
#
# ps6000 RAPID BLOCK MODE EXAMPLE
# This example opens a 6000 driver device, sets up one channel and a trigger then collects 10 block of data in rapid succession.
# This data is then plotted as mV against time in ns.

#import sys
import time
import ctypes
from picosdk.ps6000 import ps6000 as ps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc

############# Constant values
chRange=[3,3,3,3] # ranges for each channel [50mV, 50mV,..]
setCh=['setChA','setChB','setChC','setChD']
maxADC = ctypes.c_int16(32512)
microsecond=1e-6
polarity=-1

# Setting the number of sample to be collected
preTriggerSamples  = 256
postTriggerSamples = 256
maxSamples = preTriggerSamples + postTriggerSamples
print('max samples',maxSamples)
timebase=2 # 1.25GSPS
windowSize=120
startTime=preTriggerSamples
stopTime=startTime+windowSize
autoTriggerMilliseconds = 20000

nev     =100
thr_mV  =10
runMode =0
nperplot=10

daqStartTime=0
daqEndTime  =0

# Create chandle and status ready for use
status = {}
chandle = ctypes.c_int16()
connected=False
init=False

timeIntervalns = ctypes.c_float()
# Creates converted types maxSamples
cmaxSamples = ctypes.c_int32(maxSamples)
fname=''
ofile={}
dataToSave={}
read_ch_en=[True,True,True,True]
trig_ch_en=[False,False,True,True]

#### Number of points for moving average
numAve=5

def set_params(var0,var1,var2,var3,var4,var5):
    global nev
    global thr_mV
    global runMode
    global nperplot
    global fname
    nev     = var0
    thr_mV  = var1
    if (thr_mV>500):
        for ch in range(4):
            chRange[ch]=7
    runMode = var2
    fname=var3
    nperplot = int(nev/10)+1 ## to show 10 waveforms per 1 run
    print('Number of events to be collected: ',nev)

def open_scope():
    global status
    global chandle
    global connected
    # Opens the device/s
    if (connected==False):
        status["openunit"] = ps.ps6000OpenUnit(ctypes.byref(chandle), None)
        assert_pico_ok(status["openunit"])
        connected=True
    
    # Displays the serial number and handle
    print(chandle.value)

def channel_init(channel):
    global status
    global chandle
    print('Init ch',channel)
    # Set up channel A
    # handle = chandle
    # channel = ps6000_CHANNEL_A = 0
    # enabled = 1
    # coupling type = ps6000_DC_50R = 2
    # range = ps6000_50MV = 3
    # analogue offset = 0 V
    Set=setCh[channel]
    ch_range=chRange[channel]
    coupling=ps.PS6000_COUPLING["PS6000_DC_50R"]
    status[Set] = ps.ps6000SetChannel(chandle, channel, 1, coupling, ch_range, 0, 0)
    assert_pico_ok(status[Set])
    return True

def set_advancedTrigger(value,chan_en):
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
            #dirs[ch]   = ps.PS6000_THRESHOLD_DIRECTION["PS6000_INSIDE"]
            if polarity>0:
                dirs[ch]   = ps.PS6000_THRESHOLD_DIRECTION["PS6000_ABOVE"]
            else:
                dirs[ch]   = ps.PS6000_THRESHOLD_DIRECTION["PS6000_BELOW"]
            nUseCh = nUseCh+1
    triggerConditions = ps.PS6000_TRIGGER_CONDITIONS(states[0],states[1],states[2],states[3],
                                                     ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
                                                     ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"],
                                                     #ps.PS6000_TRIGGER_STATE["PS6000_CONDITION_DONT_CARE"])
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
    ch_range=chRange[0]
    threshold = mV2adc(value, ch_range, maxADC)
    if (ch_range==2):
        maxthreshold = mV2adc(50, ch_range, maxADC)
    elif (ch_range==3):
        maxthreshold = mV2adc(100, ch_range, maxADC)
    elif (ch_range==7):
        maxthreshold = mV2adc(2000, ch_range, maxADC)
    #hysteresis = mV2adc((value * 0.02), ch_range, maxADC)
    hysteresis = 0
    print('threshold=',value,', (', threshold,' in COUNT)')
    nChannelProperties = 0
    auxOutputEnable = 0
    ### Make an empty array of TRIGGER_CHANNEL_PROPERTIES with a length of "nUseCh"
    channelProperties=(ps.PS6000_TRIGGER_CHANNEL_PROPERTIES *nUseCh)()
    thre0 = min(threshold,maxthreshold)
    thre1 = max(threshold,maxthreshold)
    for ch in range(4):
        if(chan_en[ch]==False):continue
        print('### ',ch)
        mode = ps.PS6000_THRESHOLD_MODE["PS6000_LEVEL"]
        #mode = ps.PS6000_THRESHOLD_MODE["PS6000_WINDOW"]
        channelProperties[nChannelProperties] = ps.PS6000_TRIGGER_CHANNEL_PROPERTIES(polarity*thre0,
        #channelProperties[nChannelProperties] = ps.PS6000_TRIGGER_CHANNEL_PROPERTIES(threshold,
                                                                 hysteresis,
                                                                 polarity*thre1,
                                                                 #-threshold,
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
    upper = 220 # x(time interval) (0.8ns now)
    lower = 30
    ptype = ps.PS6000_PULSE_WIDTH_TYPE["PS6000_PW_TYPE_GREATER_THAN"]
    #ptype = ps.PS6000_PULSE_WIDTH_TYPE["PS6000_PW_TYPE_IN_RANGE"]
    #ptype = ps.PS6000_PULSE_WIDTH_TYPE["PS6000_PW_TYPE_NONE"]
    status["setPulseWidthQualifier"] = ps.ps6000SetPulseWidthQualifier(chandle, ctypes.byref(pwqConditions), nPwqConditions, direction, lower, upper, ptype)
    assert_pico_ok(status["setPulseWidthQualifier"])

###### Simple threshold trigger
def set_simpleTrigger(value,channel,rise): # value= threhosld in mV,channel=source channel,dir=direction
    global status
    global chandle
    direction = ps.PS6000_THRESHOLD_DIRECTION["PS6000_FALLING"]
    if rise==True:direction = ps.PS6000_THRESHOLD_DIRECTION["PS6000_RISING"]
    threshold  = mV2adc(value, chRange[channel], maxADC)
    Set='trigger'
    print('threshold=',value,', (', threshold,' in COUNT)')
    status["trigger"] = ps.ps6000SetSimpleTrigger(chandle, 1, channel, threshold, direction, 0, autoTriggerMilliseconds)
    assert_pico_ok(status[Set])

def set_timebase(base):
    # Gets timebase innfomation
    # Handle = chandle
    # Timebase = 2 = timebase
    # Nosample = maxSamples
    # TimeIntervalNanoseconds = ctypes.byref(timeIntervalns)
    # MaxSamples = ctypes.byref(returnedMaxSamples)
    # Segement index = 0
    global timebase
    global timeIntervalns
    timebase = base
    returnedMaxSamples = ctypes.c_int16()
    status["GetTimebase"] = ps.ps6000GetTimebase2(chandle, timebase, maxSamples, ctypes.byref(timeIntervalns), 1, ctypes.byref(returnedMaxSamples), 0)
    assert_pico_ok(status["GetTimebase"])

def get_single_event():
    global status
    global cmaxSamples
    # Creates a overlow location for data
    overflow = ctypes.c_int16()

    # Handle = Chandle
    # nSegments = 1
    # nMaxSamples = ctypes.byref(cmaxSamples)
    status["MemorySegments"] = ps.ps6000MemorySegments(chandle, 1, ctypes.byref(cmaxSamples))
    assert_pico_ok(status["MemorySegments"])
    
    # sets number of captures
    status["SetNoOfCaptures"] = ps.ps6000SetNoOfCaptures(chandle, 1)
    assert_pico_ok(status["SetNoOfCaptures"])
    
    # Starts the block capture
    # Handle = chandle
    # Number of prTriggerSamples
    # Number of postTriggerSamples
    # Timebase = 2 = 4ns (see Programmer's guide for more information on timebases)
    # time indisposed ms = None (This is not needed within the example)
    # segment index = 0
    # LpRead = None
    # pParameter = None
    status["runBlock"] = ps.ps6000RunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, 0, None, 0, None, None)
    assert_pico_ok(status["runBlock"])
    
    # Checks data collection to finish the capture
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps6000IsReady(chandle, ctypes.byref(ready))

    # Create buffers ready for assigning pointers for data collection
    bufferMax=[[],[],[],[]]
    bufferMin=[[],[],[],[]]
    status_str=["setDataBuffersA","setDataBuffersB","setDataBuffersC","setDataBuffersD"]
    for ch in range(4):
        if read_ch_en[ch]==True:
            bufferMax[ch] = (ctypes.c_int16 * maxSamples)()
            bufferMin[ch] = (ctypes.c_int16 * maxSamples)() # used for downsampling which isn't in the scope of this example
            # Setting the data buffer location for data collection from channel [ch]
            status[status_str[ch]] = ps.ps6000SetDataBuffers(chandle, ch, ctypes.byref(bufferMax[ch]), ctypes.byref(bufferMin[ch]), maxSamples, 0)
            assert_pico_ok(status[status_str[ch]])
    
    # Handle = chandle
    # noOfSamples = ctypes.byref(cmaxSamples)
    # start index = 0
    # ToSegmentIndex = 0
    # DownSampleRatio = 1
    # DownSampleRatioMode = 0
    # Overflow = ctypes.byref(overflow)
    status["getValues"] = ps.ps6000GetValues(chandle, 0, ctypes.byref(cmaxSamples), 1, 0, 0, ctypes.byref(overflow))
    assert_pico_ok(status["getValues"])
    
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
        if read_ch_en[ch]==True:
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

def analyse_and_plot_data(data,figname):
    # Plots the data from channel A onto a graph
    global dataToSave

    base=np.array([])
    wfrms=np.array([])
    vmax=np.array([])
    charge=np.array([])
    fig = plt.figure(figsize=(15,8))
    gs  = gridspec.GridSpec(3,3)
    ax1 = plt.subplot(gs[0,:])
    ax2 = plt.subplot(gs[1,:])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[2,1])
    ax5 = plt.subplot(gs[2,2])

    timeX = np.linspace(0, (cmaxSamples.value) * timeIntervalns.value, cmaxSamples.value)
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
            if read_ch_en[ch]==False: continue
            adc2mVChMax=np.array(adc2mV(data[i][ch], chRange[ch], maxADC))
            #avwf = polarity * running_mean(adc2mVChMax[:],numAve)
            avwf = polarity * adc2mVChMax ### Skip averaging to speed up the daq..
            baseline =avwf[:windowSize].mean()
            chargeTmp=(avwf[startTime:stopTime].sum()-baseline*windowSize)
            base  = np.append(base, baseline)                     # mean value of the baseline
            wfrms = np.append(wfrms, avwf[:windowSize].std())     # standard deviation
            vmax  = np.append(vmax, avwf[startTime2:stopTime2].max()-baseline) # maximum voltage within the time window
            charge= np.append(charge, chargeTmp*float(timeIntervalns.value))   # integrated charge
            if len(waveforms)==0:
                waveforms=np.transpose(adc2mVChMax)
            else:
                waveforms=np.vstack([waveforms,np.transpose(adc2mVChMax)])
            if ch>0:continue #### plot only chA for now...
            if (i%nperplot)!=0:continue
            ax1.plot(timeX, adc2mVChMax[:]-baseline)
            #ax2.plot(timeX[numAve-1:], avwf-baseline) if averaging is "ON"
            ax2.plot(timeX, avwf-baseline)

    dataToSave.append(waveforms)

    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_xlim(-1,(cmaxSamples.value) * timeIntervalns.value + 1)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.set_xlim(-1,(cmaxSamples.value) * timeIntervalns.value + 1)
    
    nbins=100
    
    ymax=0.15*float(nev)
    xbins=np.linspace(0,0.9,nbins)
    ax3.hist(wfrms,bins=xbins)
    ax3.set_ylim(0.8,ymax)
    ax3.set_title('Pedestal RMS')
    ax3.text(+0.5,0.4*ymax,r'$\mu=$'+f'{wfrms.mean():.2f}'+' mV',fontsize=12)
    ax3.set_xlabel('rms (mv)')
     
    xbins=np.linspace(0,50,nbins)
    ax4.hist(vmax,bins=xbins,density=True)
    ax4.text(25,0.5*ymax,r'$\mu=$'+f'{vmax.mean():.2f}'+' mV',fontsize=12)
    ax4.set_ylim(0,1)
    ax4.set_yscale('log')
    ax4.set_title('Max Peak value')
    ax4.set_xlabel('Max Voltage (mv)')
    
    ymax=0.08*float(nev)
    xbins=np.linspace(-50,50,nbins)
    ax5.hist(charge,bins=xbins)
    ax5.set_ylim(0.8,ymax)
    ax5.set_yscale('log')
    ax5.text(22,0.4*ymax,r'$\mu=$'+f'{charge.mean():.2f}',fontsize=12)
    ax5.set_title('Integrated charge')
    ax5.set_xlabel('Charge (mv*ns)')
    
    fig.savefig('/home/comet/Desktop/'+figname)
    plt.close(fig)

### Initialise channel A & B
def init_daq():
    global init
    global polarity
    global trig_ch_en
    if runMode==0: trig_ch_en=[False,False,True,False] ### !!Temporary
    if init==False:
        open_scope()
        for ch in range(4):
            if read_ch_en[ch]==True or trig_ch_en[ch]==True:
                channel_init(ch)
        ### pedestal run
        if   runMode==0:
            trigCh=-1
            for ch in range(4):
                if trig_ch_en[ch]==True:
                    trigCh=ch
                    break 
            if trigCh<0 or trigCh>3:
                print("Illegal Trigger Channel: ",trigCh)
                exit()
            print('Trigger ch: ',trigCh)
            set_simpleTrigger(thr_mV,trigCh,True)
        elif runMode==1: # negative polarity for SiPM signals
            polarity = -1
            ch_en=trig_ch_en
            set_advancedTrigger(thr_mV,ch_en)
        elif runMode==2: # positive polarity for other tests
            polarity = +1
            ch_en=trig_ch_en
            set_advancedTrigger(thr_mV,ch_en)
        init=True

def run_daq(sub):
    set_timebase(2) ## 1.25GSPS
    data=[]
    global ofile
    global dataToSave
    global daqStartTime
    global daqEndTime
    fname_sub='/home/comet/work/pico/data/'+fname+'_'+str(sub)+'.npy'
    ofile=open(fname_sub,"wb")
    #print('time interval = ',timeIntervalns.value)
    print('integration from ',startTime*timeIntervalns.value,' to ',stopTime*timeIntervalns.value,' [ns]')
    daqStartTime=time.time()
    for iev in range(nev):
        #adc2mVData=get_single_event()
        #data.append(adc2mVData)
        rawdata=get_single_event()
        data.append(rawdata)
        time.sleep(100*microsecond) ### Trigger rate is limited to 10kHz here.
    daqEndTime=time.time()

    print('Trigger rate = ', float(nev)/(daqEndTime-daqStartTime), ' Hz'), 
    # Stops the scope
    # Handle = chandle
    status["stop"] = ps.ps6000Stop(chandle)
    assert_pico_ok(status["stop"])
    dataToSave={} #Init
    analyse_and_plot_data(data,'figA.png')
    np.save(ofile,dataToSave,allow_pickle=True)
    ofile.close

def close():
    # Closes the unit
    # Handle = chandle
    status["close"] = ps.ps6000CloseUnit(chandle)
    assert_pico_ok(status["close"])
    
    # Displays the staus returns
    print(status)

