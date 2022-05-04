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
            charges.append(self.getHeight(i))
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

#Global variables
SigLower = 0
SigUpper = 400
BaseUpper = 100
Polarity  = -1

### Flags for each filtering
MovAvF = 0
FreqF  = 0
BaseF  = 0

CutoffFreq = 400 ## FFT cut-off frequency in MHz
MNumber    = 20  ## moving average filter number

NBins = 100 #Histogram bins used 
RangeUpper    = 100 #Upper limit of histograms 
RangeLower    = 0 #Lower limit of histograms 
TimeUpper     = 100 #Upper limit of histograms 
TimeLower     = 0 #Lower limit of histograms 
TimeBins      = 50
TimeScale     = 0.8 # It's 4ns/sample for PS3000

#Set a baseline RMS cut off to remove noise
RMS_Cut = 3.0 #mV (based on plotting RMS values for baseline window [:50])

ConstantFraction=0.15
PeakThreshold=10

##def Initialise(): # to be implemented
#### Basic functions

#### Set the binnings and range for PlotHistogram
def SetBins(nbins,lower,upper):
    global NBins
    global RangeUpper
    global RangeLower
    if (nbins<1):
        print("ERROR! negative bin size is assigned")
        return -1
    if (upper<lower):
        print("ERROR! upper limit must be greater than lower limit")
        return -1
    NBins = nbins
    RangeUpper    = upper
    RangeLower    = lower
    return 0

def SetPolarity(val):
    global Polarity
    Polarity = val

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
    TimeBins  = int((TimeUpper-TimeLower)/2)

def SetPeakThreshold(val):
    global PeakThreshold
    PeakThreshold = val

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
    Copy[np.abs(samplefreq)>CutoffFreq*1e6]=0
    
    filteredsig=fftpack.ifft(Copy)
    Signal = filteredsig.real

    return Signal

def BaselineFilter(Signal):
    base = np.mean(Signal[:BaseUpper])
    return Signal-base

def MovAvFilter(Signal):
    #Apply moving average filter to waveform
    #MNumber #moving average filter number
    CutEdges = MNumber-1 #cut edges off to account for filter effects
    
    moveavefilt=np.cumsum(np.insert(Signal,0,0))
    Signal=(moveavefilt[MNumber:]-moveavefilt[:-MNumber])/float(MNumber)
    Signal = Signal[CutEdges:Signal.size-CutEdges]
    
    return Signal
 
def PlotWaveformsFromAFile(FName):
    #Plot all waveforms from a given file
    if (LoadFile(FName)==False):ErrorExit("DecodeChannels()")

    Waveforms,SumWaveforms = DecodeChannels(FName)  
    for ch in range(NCh): 
        plt.figure()
        for i in range(len(Waveforms[ch])):
            #plt.figure()
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (mV)")
            plt.title("Ch "+str(ch))
            plt.plot(Waveforms[ch][i],alpha=0.1)
       
    if NCh>1: 
        plt.figure()
        plt.xlabel("Time (a.u.)")
        plt.ylabel("Voltage (mV)")
        plt.title("Summed Waveforms")
        for i in range(len(SumWaveforms)):
            plt.plot(SumWaveforms[i],alpha=0.1)
                

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
    
    ## Make a summed waveform
    for i in range(NCh):
        ChDecoded[i] = np.array(ChDecoded[i])
    ChSum = np.copy(ChDecoded[0]) #Copy Ch A
    for i in range(1,NCh):
        ChSum+=ChDecoded[i] #Add Ch B,C,D
    
    return ChDecoded, ChSum

### Enabling Moving Average Filter
def EnableMovingAverageFilter(num):
    global MovAvF
    global MNumber
    MovAvF  =  1
    MNumber = num
    return

### Enabling FFT-based Filter
def EnableFFTFilter(freq):
    global FreqF
    global CutoffFreq
    CutoffFreq = freq
    FreqF = 1
    return

### Enabling Baseline subtraction
def EnableBaselineFilter():
    global BaseF
    BaseF = 1
    return

### Get the index where waveform exceeds the certain threshold
### Put the waveform, Wf[AnalysisWindow:PeakIdx+1] as Wf
def GetEdgeTime(Wf,constantFraction=True):
    threVal = PeakThreshold
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
    if(BaseF ==1): Signal = BaselineFilter(Signal)
    Signal = Polarity*Signal
    ChT = np.linspace(0,len(Signal)*TimeScale,len(Signal)) #All channels have a dt = 0.8 ns
    
    #Calculate RMS of baseline area before signal window
    RMS = np.std(Signal[:BaseUpper])
    
    #Extract output analysis parameter from waveform
    PeakVal   = np.max(Signal[SigLower:SigUpper])
    PeakIndex = np.argmax(Signal[SigLower:SigUpper])+SigLower
    if(PeakVal>PeakThreshold):
        EdgeTime = GetEdgeTime(Signal[SigLower:PeakIndex+1])+SigLower*TimeScale
    else:
        EdgeTime = -1
    ChargeVal = simps(Signal,ChT) # scipy integration function
    
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
  
def PlotHistogram(Data,RangeLower,RangeUpper,NBins,String,strData): #pdist,threshold,subplot,Ped_Peak,SP_Peak,uPed,uSP):
    #Take collected channel data from all files to be analysed and plot histogram
    #Data for a given channel
    #RangeUpper,RangeLower = range of histogram
    #NBins = number of bins
    #String = title string on plot
    
    colour = 'purple'
    alpha = 0.5
    
    plt.figure()
    CurrentN,CurrentBins,_=plt.hist(Data,range=[RangeLower,RangeUpper],bins=NBins,color=colour,alpha=alpha)
    plt.title(String)
    plt.xlabel(strData)
    plt.ylabel("Count")

    return CurrentBins , CurrentN

def AnalyseSingleFile(FName,ChOutputs,ChSumOut):
    global FileLoaded
    #Takes a file path and analyses all waveforms in the file
    #Returns an output with form [[PeakIndex, PeakValue, ChargeValue, BaselineRMS],...,[]] containing info for each waveform
        #PeakIndex   = Index within file of analysed value (i.e. index of signal peak)
        #PeakValue   = Analysed value (i.e. peak value or integrated charge)
        #ChargeValue = Analysed value (i.e. peak value or integrated charge)
        #BaselineRMS = RMS (standard deviation) of Baseline
    #This is returned for each channel separately
    if (FileLoaded==False): LoadFile(FName)
    
    Waveforms,SumWaveforms = DecodeChannels(FName)
    
    NWaveforms = len(Waveforms[0]) #All channels have same number of waveforms
    TRate = (HeaderInfo[0][4]/(HeaderInfo[0][3]-HeaderInfo[0][2]))
    
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
        ChSumOut.append(ProcessAWaveform('Sum',SumWaveforms[i]))
    #print(ChSumOut)
    ## Prepare to read the next file
    FileLoaded = False
    return TRate

def AnalyseFolder(FPath,PlotFlag=False):
    #Analyse all data files in folder located at FPath
    #PlotFlag is an option to output histograms or not
    
    MeanTR = 0
    
    FList = FileList(FPath)
    TriggerRates = []
   
    FileOutputs=[[],[],[],[]]  # 4channels
    SumOutputs=[]
    for i in range(len(FList)):
        print("Analysing file:",FList[i][len(FPath)-1:])
        TRate = AnalyseSingleFile(FList[i],FileOutputs,SumOutputs)
        TriggerRates.append(TRate)
        print("Trigger rate (Hz) = ",TRate)

    TriggerRates=np.array(TriggerRates)   
    MeanTR = np.mean(TriggerRates)
    
    ChHistData=[]
    for ch in range(NCh): 
        dataArray = ExtractWfInfo(FileOutputs[ch])
        print(dataArray)
        heightArray = np.array(dataArray.getHeightArray(),dtype=np.float)
        print(heightArray)
        nBins, vals = PlotHistogram(heightArray,RangeLower,RangeUpper,NBins,str(dataArray.getChannel(0)),
                "Peak height [mV]")
        ChHistData.append(nBins)
        ChHistData.append(vals)

        #Added Feb 27 2022 - plotting time of peak in array (Ch B will have garbage so ignore)
        #timearray = np.array(dataArray.getPeakIndexArray(),dtype=np.float) #
        timearray = np.array(dataArray.getEdgeTimeArray(),dtype=np.float) #
        nBinsT, valsT = PlotHistogram(timearray,TimeScale*TimeLower,TimeScale*TimeUpper,TimeBins,str(dataArray.getChannel(0)),
                #"Peak Time (ns)")
                "Edge Time (ns)")
        #ChHistData.append(nBinsT)
        #ChHistData.append(valsT)        
        TimePeak = nBinsT[np.argmax(valsT)]
        print("EdgeTime = %s ns" %(TimePeak))

    #Plot summed channel histogram
    dataArray = ExtractWfInfo(SumOutputs)
    print(dataArray)
    heightArray = np.array(dataArray.getHeightArray(),dtype=np.float)
    nBins, vals = PlotHistogram(heightArray,4*RangeLower,4*RangeUpper,int(2*NBins),str(dataArray.getChannel(0)),"Peak height [mV]") 
    ChHistData.append(nBins)
    ChHistData.append(vals)
    
    # Return data in form NCh, MeanTR, [ChABin,ChAN,...]
    ChHistData = np.array(ChHistData)
    return NCh, MeanTR, ChHistData 


#### Following Specific Analysis Function should be moved to outside
#### Leave this for an example, but please don't keep modifying and using this..
def CosmicSr90Analysis():
    #Function to determine Sr90 spectrum from datasets of cosmic rays and Sr90 + cosmic rays
    
    #Analyse cosmic ray data set - note this is not a purely min ionising cosmic data set
    #FolderPath=r'/home/comet/work/pico/Oct15Cosmic//' 
    #nch,CosmicTR,CosmicHistData = AnalyseFolder(FolderPath,True)
    #CosmicBins = CosmicHistData[8]
    #CosmicN = CosmicHistData[9]
    
    #Analyse strontium data set
    FolderPath = r'/home/comet/work/data/Dec13_LargeScint_Cosmic_SmallTrigNearFibre_AUXTrig150mV_Vb42_MinDist'
    nch,SrTR,SrHistData = AnalyseFolder(FolderPath,True)
    SrBins = SrHistData[8]
    SrN = SrHistData[9]
    
    #Subtract cosmic spectrum from strontium + cosmic spectrum
    #nCosmicN = CosmicTR*CosmicN/np.sum(CosmicN)
    #print("Mean cosmic trigger rate = ", CosmicTR)
    #nSrN = SrTR*SrN/np.sum(SrN)
    print("Mean cosmic + Sr90 trigger rate = ", SrTR)
    #SrSpectrum = nSrN-nCosmicN    

    #Plot histogram using bar plot
    #plt.figure()
    #plt.bar(SrBins[:-1],SrSpectrum,width=SrBins[1]-SrBins[0], color='blue') 
    #plt.title("Reverse Strontium Spectrum")
    #plt.xlabel(XString)
    #plt.ylabel("Count")
    
    #Determine endpoint of spectrum?
    return

#################################################################################
### Following commands are supposed to be done in your own analysis script.
###  Please refer to myAna.py
###FolderPath=r'/home/comet/work/data/Dec10_TrigScint_Sr90Col_SelfTrig50mV_Vb42_MinDist' 
###PlotRMS(FolderPath,"Sr90")
###PlotWaveformsFromAFile(FolderPath+"ScintTestOct15_Sr90_T9mV_99.npy")
###CosmicSr90Analysis()
###plt.show()
