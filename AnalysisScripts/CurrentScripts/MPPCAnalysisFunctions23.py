# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:29:13 2021

Reorganised MPPC Functions

@author: smdek2
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import fftpack
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import simps

import glob 
import math
import sys
import os

class WfInfo(object):
    'Base class to store the information extracted from each waveform'
    ## ch     : channel
    ## peakIdx: Peak index
    ## height : Peak height
    ## charge : Integrated charge
    ## rms    : baseline RMS
    ## edgeTime: Pulse edge timing
    def __init__(self, ch, peakIdx, height, charge, rms, edgeTime):
        self.ch = ch
        self.peakIdx = peakIdx
        self.height  = height
        self.charge  = charge
        self.rms     = rms
        self.edgeTime= edgeTime
    def ch(self):
        return self.ch
    def peakIdx(self):
        return self.peakIdx
    def height(self):
        return self.height
    def charge(self):
        return self.charge
    def rms(self):
        return self.rms
    def edgeTime(self):
        return self.edgeTime

class ExtractWfInfo():
    def __init__(self, wfList):
        self.wfList = list(wfList)
    def getWfAt(self,i):
        wf = self.wfList[i]
        #print(wf)
        return wf
    def getChannel(self,i):
        return self.getWfAt(i).ch
    def getHeight(self,i):
        return self.getWfAt(i).height
    def getCharge(self,i):
        return self.getWfAt(i).charge
    #Added Feb 27 2022
    def getPeakIndex(self,i):
        return self.getWfAt(i).peakIdx
    def getEdgeTime(self,i):
        return self.getWfAt(i).edgeTime

    def getHeightArray(self):
        heights=[]
        for i in range(len(self.wfList)):
            heights.append(self.getHeight(i))
        return heights
    def getChargeArray(self):
        charges=[]
        for i in range(len(self.wfList)):
            charges.append(self.getCharge(i))
        return charges
    #Added Feb 27 2022
    def getPeakIndexArray(self):
        ptimes=[]
        for i in range(len(self.wfList)):
            ptimes.append(self.getPeakIndex(i))
        return ptimes
    def getEdgeTimeArray(self):
        etimes=[]
        for i in range(len(self.wfList)):
            etimes.append(self.getEdgeTime(i))
        return etimes

FileLoaded = False
FileData   = []
HeaderInfo = []
WfData     = []
RemoveNoisyEvent=False
NCh   = 0
Nevents = 0

#Global variables
SigLower = [0,0,0,0]
SigUpper = [400,400,400,400]
BaseUpper = [100,100,100,100]
Polarity  = [1,1,1,1,1]

### Flags for each filtering
MovAvF = 0
FreqF  = 0
BaseF  = 0
TrigF  = 0
DiffF  = 0


UpperCutoffFreq = 400 ## FFT cut-off frequency in MHz (high)
LowerCutoffFreq = 400 ##FFT cut-off frequency in MHz (low)
MNumber    = 20  ## moving average filter number
TrigCutOffLow = 50 ## trigger cut value low
TrigCutOffHigh = 150 ## trigger cut value high
TrigCh     = 1   ## trigger channel
DiffN = 1 ## diff filter n

NBins = [100,100,100,100] #Histogram bins used 
RangeUpper    = [100,100,100,100] #Upper limit of histograms 
RangeLower    = [100,100,100,100] #Lower limit of histograms 
TimeUpper     = 100 #Upper limit of histograms 
TimeLower     = 0 #Lower limit of histograms 
TimeBins      = 50
TimeScale     = 0.8 # It's 4ns/sample for PS3000

#Set a baseline RMS cut off to remove noise
RMS_Cut = 3.0 #mV (based on plotting RMS values for baseline window [:50])

ConstantFraction=0.15
PeakThreshold=[10,10,10,10,40]

##def Initialise(): # to be implemented
#### Basic functions

#### Set the binnings and range for PlotHistogram
def SetBins(BinSize,lower,upper):
    global NBins
    global RangeUpper
    global RangeLower
    #for i in nbins: 
    #    if (i<1): 
    #        print("ERROR! negative bin size is assigned")
    #        return -1
    #if (upper<lower):
    #    print("ERROR! upper limit must be greater than lower limit")
    #    return -1

    NBins = np.rint((np.array(upper)-np.array(lower))/np.array(BinSize)).astype(int)
    print("BINS")
    print(NBins)
    RangeUpper    = upper
    RangeLower    = lower
    return 0

def SetPolarity(vals):
    global Polarity
    Polarity = np.array(vals).astype(int)

def SetTimeScale(val):
    global TimeScale
    TimeScale = val

def SetSignalWindow(sigL,sigU,baseU=100):
    global SigLower
    global SigUpper
    global BaseUpper
    global TimeLower
    global TimeUpper
    
    SigLower  = sigL  - (MNumber-1)
    SigUpper  = sigU  - (MNumber-1)
    BaseUpper = baseU - (MNumber-1)
    TimeLower = SigLower - 1
    TimeUpper = SigUpper + 1
    #TimeBins  = int((TimeUpper-TimeLower)/2)

def SetPeakThreshold(vals):
    global PeakThreshold
    PeakThreshold = np.array(vals).astype(float)

def SetConstantFraction(val):
    global ConstantFraction
    ConstantFraction = val

def SetRMSCut(val):
    global RMS_Cut
    RMS_Cut = val

def ErrorExit(String):
    #Exit program with string saying where
    print("Error at ",String)
    sys.exit()
    return

def FFTFilter(Data):
    #Apply fourier transform to a signal and cut all frequency components above a threshold    
    Samples = Data.size 
    dt = 0.8*10**-9
    fftV = fftpack.fft(Data)   
    samplefreq=fftpack.fftfreq(Samples, dt)
    Copy = fftV.copy()
    Copy[np.abs(samplefreq)>UpperCutoffFreq*1e6]=0
    if(LowerCutoffFreq!=UpperCutoffFreq):
        Copy[np.abs(samplefreq)<LowerCutoffFreq*1e6]=0

    filteredsig=fftpack.ifft(Copy)
    Signal = filteredsig.real

    return Signal

def DiffFilter(Data):
    dY = Data[DiffN:]-Data[:-DiffN] 
    return dY

def BaselineFilter(Ch, Signal):
    base = np.mean(Signal[:BaseUpper[Ch]])
    return Signal-base

def MovAvFilter(Signal):
    #Apply moving average filter to waveform
    #MNumber #moving average filter number
    CutEdges = MNumber-1 #cut edges off to account for filter effects
    
    moveavefilt=np.cumsum(np.insert(Signal,0,0))
    Signal=(moveavefilt[MNumber:]-moveavefilt[:-MNumber])/float(MNumber)
    Signal = Signal[CutEdges:Signal.size-CutEdges]
    
    return Signal
 
def PlotWaveformsFromAFile(FName,plotch=-1,start=-1,end=-1):
    #Plot all waveforms from a given file

    #Plot all waveforms from a given file
    if (LoadFile(FName)==False):ErrorExit("DecodeChannels()")
    AnalyseFlag=1
    Waveforms = DecodeChannels(FName)  
    for ch in range(NCh): 
        if (plotch>0 and ch!=plotch): continue
        plt.figure()
        plt.figure(figsize=(16,8),dpi=80)
        for i in range(len(Waveforms[ch])):
            #plt.figure()
            if (start>=0 and i<start): continue
            if (end>=0   and i>end)  : break
            
            if(AnalyseFlag==1):
                Signal = Waveforms[ch][i]
                #plt.plot(Signal,alpha=0.1,color='b',label='Raw')
                #if(FreqF ==1): Signal = FFTFilter(Signal)   ### Move this before the moving average filter
                #plt.plot(Signal,alpha=0.1,label='FFT')
                #if(DiffF==1): Signal = DiffFilter(Signal)
                #plt.plot(Signal,alpha = 0.5, label = 'Diff')
                if(MovAvF==1): Signal = MovAvFilter(Signal)
                #if(FreqF ==1): Signal = FFTFilter(Signal)   ### Move this before the moving average filter
                #plt.plot(Signal,alpha=0.1,color='r',label='MA')
                #if(DiffF==1): Signal = DiffFilter(Signal)
                plt.plot(Signal,alpha = 0.1,color='g', label = 'Diff')
                #if(BaseF ==1): Signal = BaselineFilter(ch,Signal)
                #Signal = Polarity[int(ch)]*Signal
                #ChT = np.linspace(0,len(Signal)*TimeScale,len(Signal)) #All channels have a dt = 0.8 ns
                BU = BaseUpper[ch]
                SU = SigUpper[ch]
                SL = SigLower[ch]

                #Calculate RMS of baseline area before signal window
                RMS = np.std(Signal[:BU])
    
                #Extract output analysis parameter from waveform
                PeakVal   = np.max(Signal[SL:SU])
                PeakIndex = np.argmax(Signal[SL:SU])+SL
                #if(PeakVal>PeakThreshold[int(Ch)]):
                #    EdgeTime = GetEdgeTime(Signal[SL:PeakIndex+1],Ch)+SL*TimeScale
                #else:
                #    EdgeTime = -1
 
                #ChargeVal = simps(Signal[SL:SU],ChT[SL:SU]) # scipy integration function
     
                #Append outputs
          
            if(AnalyseFlag==1): 
                PlotSignal = Signal
                #plt.plot(Signal, alpha=0.9)
            else: 
                PlotSignal = Waveforms[ch][i]
                plt.plot(PlotSignal,alpha=0.1)
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (mV)")
            plt.ylim([-150,-50])
            plt.xlim([150,250])
            plt.legend()
            plt.title("Ch "+str(ch))

        plt.savefig("Channel_"+str(ch)+".png")
       
                

def FileList(FPath):
    #For a given folder path, return the files
    FileListNew = []
    FilePaths = str(FPath)+'/*.npy' #file paths of every .npy file
    FileList=glob.glob(FilePaths,recursive=True)
    FileList.sort(key=os.path.getmtime)
    for f in FileList:
        size = os.path.getsize(f)
        if (size==0): continue
        FileListNew.append(f)

    #print(size)
    #if(size==0):FileList = FileList[:-1] 
    return FileListNew

def LoadFile(FName):
    #Returns number of channels
    global FileData
    global HeaderInfo
    global NCh
    global WfData
    global FileLoaded
    FileData   = np.array(np.load(FName,allow_pickle=True))
    HeaderInfo = np.array([FileData[:6]])
    WfData     = FileData[6]
    NCh        = int(HeaderInfo[0][1][0]+HeaderInfo[0][1][1]+HeaderInfo[0][1][2]+HeaderInfo[0][1][3])
    FileLoaded = True
    if (NCh<1 or NCh>4): return False
    return True

def DecodeChannels(FName):
    #For a given file extract and return the channel data + header info
    #Get waveforms from the npy file
    if (FileLoaded==False):
        if(LoadFile(FName)==False):ErrorExit("DecodeChannels()")
    
    ChDecoded = [[],[],[],[]] ## empty list to store the waveforms
    ## Loop over the number of recorded events 
    for i in range(int(len(WfData[:,0])/NCh)):
        ## Loop over the different channels
        for j in range(NCh):
            ChDecoded[j].append(WfData[NCh*i+j,:])
    
    return ChDecoded

### Enabling Moving Average Filter
def EnableMovingAverageFilter(num):
    global MovAvF
    global MNumber
    MovAvF  =  1
    MNumber = num
    return

### Enabling FFT-based Filter
def EnableFFTFilter(freqU,freqL):
    global FreqF
    global UpperCutoffFreq
    global LowerCutoffFreq
    UpperCutoffFreq = freqU
    LowerCutoffFreq = freqL
    FreqF = 1
    return

def EnableDiffFilter(n):
    global DiffN
    global DiffF
    DiffN = n
    DiffF = 1
    return

### Enabling Baseline subtraction
def EnableBaselineFilter():
    global BaseF
    BaseF = 1
    return

### Enabling trigger peak cut
def EnableTriggerCut(TC,TTL,TTH):
    global TrigF
    global TrigCh
    global TrigCutOffLow
    global TrigCutOffHigh
    TrigF = 1
    TrigCutOffLow = TTL
    TrigCutOffHigh = TTH
    TrigCh = TC
    return

### Compare trigger channel to a data channel and remove above threshold
def TriggerCut(Ch,ChannelData,TrigData):

    if(Ch!=TrigCh): Data = ChannelData[(TrigData <= TrigCutOffHigh) & (TrigData >= TrigCutOffLow)]
    
    elif(Ch==TrigCh): Data = TrigData[(TrigData <= TrigCutOffHigh) & (TrigData >= TrigCutOffLow)]
    
    return Data

### Get the index where waveform exceeds the certain threshold
### Put the waveform, Wf[AnalysisWindow:PeakIdx+1] as Wf
def GetEdgeTime(Wf,Ch,constantFraction=False):
    threVal = PeakThreshold[int(Ch)]
    if constantFraction==True:
        threVal = ConstantFraction*np.max(Wf)
    ret = np.where(Wf<threVal)
    if len(ret[0])==0: return -1
    #print(ret,"  ",len(ret[0]))
    idx  = ret[0][-1]
    #print(idx)
    time = ( (threVal-Wf[idx])*(idx+1) + (Wf[idx+1]-threVal)*(idx) ) / (Wf[idx+1]-Wf[idx])
    time = TimeScale*time
    return time
    
def ProcessAWaveform(Ch,Signal):
    #Extract information from a signal waveform:
    #    (Peak index, Peak value, Integrated charge, Noise RMS, Noise flag
    #Filter - If 1, apply a moving average filter
    #FreqF - If 1, applies a FFT to remove high frequency components
   
    #### Not so fure the following oder is the best or not 
    if(FreqF ==1): Signal = FFTFilter(Signal)   ### Move this before the moving average filter
    if(MovAvF==1): Signal = MovAvFilter(Signal)
    if(DiffF == 1): Signal = DiffFilter(Signal)
    if(BaseF ==1): Signal = BaselineFilter(0,Signal) 
    Signal = Polarity[int(Ch)]*Signal
    ChT = np.linspace(0,len(Signal)*TimeScale,len(Signal)) #All channels have a dt = 0.8 ns
    
    BU = BaseUpper[Ch]
    SU = SigUpper[Ch]
    SL = SigLower[Ch]

    #Calculate RMS of baseline area before signal window
    RMS = np.std(Signal[:BU])
    
    #Extract output analysis parameter from waveform
    PeakVal   = np.max(Signal[SL:SU])
    PeakIndex = np.argmax(Signal[SL:SU])+SL
    
    if(PeakVal>PeakThreshold[int(Ch)]):
            EdgeTime = GetEdgeTime(Signal[SL:PeakIndex+1],Ch)+SL*TimeScale
    else:
            EdgeTime = -1
    #EdgeTime = -1

    ChargeVal = simps(Signal[SL:SU],ChT[SL:SU]) # scipy integration function

    #Append outputs
    return WfInfo(Ch,PeakIndex,PeakVal,ChargeVal,RMS,EdgeTime)
   
def HistogramFit(x,*params):
    y = np.zeros_like(x)
    for i in range(0,len(params),3):
        mean = params[i]
        amplitude = params[i+1]
        sigma = params[i+2]
        y = y + amplitude * np.exp(-((x-mean)/sigma)**2)
    return y  
  
def PlotHistogram(Data,RL,RU,NB,String,strData,PlotFig=False): #pdist,threshold,subplot,Ped_Peak,SP_Peak,uPed,uSP):
    #Take collected channel data from all files to be analysed and plot histogram
    #Data for a given channel
    #RangeUpper,RangeLower = range of histogram
    #NBins = number of bins
    #String = title string on plot
    
    #colour = 'purple'
    alpha = 0.5
    
    if(PlotFig):plt.figure()
    #CurrentN,CurrentBins,_=plt.hist(Data,range=[RangeLower,RangeUpper],bins=NBins,color=colour,alpha=alpha)
    CurrentN,CurrentBins,_=plt.hist(Data,range=[RL,RU],bins=NB,alpha=alpha)
    if(PlotFig):
        plt.title(String)
        plt.xlabel(strData)
        plt.ylabel("Count")
    
    return CurrentBins , CurrentN

def GetNumEvents():
    return Nevents

def AnalyseSingleFile(FName,ChOutputs):
    global FileLoaded
    global Nevents
    #Takes a file path and analyses all waveforms in the file
    #Returns an output with form [[PeakIndex, PeakValue, ChargeValue, BaselineRMS],...,[]] containing info for each waveform
        #PeakIndex   = Index within file of analysed value (i.e. index of signal peak)
        #PeakValue   = Analysed value (i.e. peak value or integrated charge)
        #ChargeValue = Analysed value (i.e. peak value or integrated charge)
        #BaselineRMS = RMS (standard deviation) of Baseline
    #This is returned for each channel separately
    if (FileLoaded==False): LoadFile(FName)
    
    Waveforms = DecodeChannels(FName)
    
    NWaveforms = len(Waveforms[0]) #All channels have same number of waveforms
    TRate = (HeaderInfo[0][4]/(HeaderInfo[0][3]-HeaderInfo[0][2]))
    Nevents += NWaveforms
    
    for i in range(NWaveforms):
        NoisyEvent=False
        wfInfo=[[],[],[],[]]
        #### check each channel
        for ch in range(NCh):
            wfInfo[ch] = ProcessAWaveform(ch,Waveforms[ch][i])
            if (wfInfo[ch].rms>RMS_Cut): NoisyEvent=True

        if (RemoveNoisyEvent==True and NoisyEvent==True): continue
        for ch in range(NCh):
            ChOutputs[ch].append(wfInfo[ch])
    
    ## Prepare to read the next file
    FileLoaded = False
    return TRate

def AnalyseFolder(FPath,PlotFlag=False,start=0,end=0):
    #Analyse all data files in folder located at FPath
    #PlotFlag is an option to output histograms or not
    
    MeanTR = 0
    
    FList = FileList(FPath)
    if end==0:
        end = len(FList)
    TriggerRates = []
   
    FileOutputs=[[],[],[],[]]  # 4channels
    

    for i in range(start,end):
        print("Analysing file:",FList[i][len(FPath)-1:])
        TRate = AnalyseSingleFile(FList[i],FileOutputs)
        TriggerRates.append(TRate)
        print("Trigger rate (Hz) = ",TRate)

    TriggerRates=np.array(TriggerRates)   
    MeanTR = np.mean(TriggerRates)
    
    ChPeakData=[]
    ChTData=[]
    heightArray=[]
    timeArray=[]
    nBins = []
    for ch in range(NCh): 
        dataArray = ExtractWfInfo(FileOutputs[ch])
        print("Channel ", ch)
       
        heightArrayTmp = np.array(dataArray.getHeightArray(),dtype=np.float)
        heightArray.append(heightArrayTmp)

        timeArrayTmp = np.array(dataArray.getEdgeTimeArray(), dtype=np.float)
        timeArray.append(timeArrayTmp)
    
    for ch in range(NCh): 
        nBins, vals = PlotHistogram(heightArray[ch],RangeLower[ch],RangeUpper[ch],NBins[ch],str(dataArray.getChannel(0)), "Peak height [mV]")
          
        #Coincidence Plotting
        t_thresh = 10
        #nBins, vals = PlotHistogram(heightArray[ch][((heightArray[2]>-70)|(heightArray[3]>-70))&(((timeArray[3]-timeArray[0])<t_thresh)|((timeArray[3]-timeArray[1])<t_thresh)|((timeArray[2]-timeArray[0])<t_thresh)|((timeArray[2]-timeArray[1])<t_thresh))],RangeLower[ch],RangeUpper[ch],NBins[ch],str(dataArray.getChannel(0)), "Peak height [mV]")
        
        ChPeakData.append(nBins)
        ChPeakData.append(vals)


        nTBins, Tvals = PlotHistogram(timeArray[ch], 3000, 4000, 1000,  str(dataArray.getChannel(0)), "Edge Time [a.u.]")
    
        ChTData.append(nTBins)
        ChTData.append(Tvals)

    ChPeakData = np.array(ChPeakData)
    ChTData = np.array(ChTData)
    return NCh, TriggerRates, ChPeakData, ChTData, Nevents 
