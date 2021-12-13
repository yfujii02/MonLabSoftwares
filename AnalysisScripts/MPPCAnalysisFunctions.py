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
    def __init__(self, ch, peakIdx, height, charge, rms):
        self.ch = ch
        self.peakIdx = peakIdx
        self.height  = height
        self.charge  = charge
        self.rms     = rms
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

FileLoaded = False
FileData   = []
HeaderInfo = []
WfData     = []
RemoveNoisyEvent=True
NCh   = 0

#Global variables
SigLower = 0
SigUpper = 400
SigLowerRMS = 100

NBins = 100 #Histogram bins used 
RangeUpper    = 100 #Upper limit of histograms 
RangeLower    = 0 #Lower limit of histograms 

#Set a baseline RMS cut off to remove noise
RMS_Cut = 3.0 #mV (based on plotting RMS values for baseline window [:50])

##def Initialise(): # to be implemented

#### Set the binnings and range for PlotHistogram
def SetBins(nbins,upper,lower):
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

#### Basic functions
def SetRMSCut(val):
    global RMS_Cut
    RMS_Cut = val

def SetSignalWindow(sigL,sigU):
    global SigLower
    global SigUpper
    
    SigLower = sigL
    SigUpper = sigU

def ErrorExit(String):
    #Exit program with string saying where
    print("Error at ",String)
    sys.exit()
    return

def FFT(Data):
    #Apply fourier transform to a signal and cut all frequency components above a threshold    
    Samples = Data.size 
    dt = 0.8*10**-9
    ThresholdFreq = 0.4e8
    fftV = fftpack.fft(Data)   
    samplefreq=fftpack.fftfreq(Samples, dt)
    Copy = fftV.copy()
    Copy[np.abs(samplefreq)>ThresholdFreq]=0
    
    filteredsig=fftpack.ifft(Copy)
    Signal = filteredsig.real

    return Signal

def MovAvFilter(Signal):
    #Apply moving average filter to waveform
    MNumber = 20 #moving average filter number
    CutEdges = 5 #cut edges off to account for filter effects
    
    moveavefilt=np.cumsum(np.insert(Signal,0,0))
    Signal=(moveavefilt[MNumber:]-moveavefilt[:-MNumber])/float(MNumber)
    Signal = Signal[CutEdges:Signal.size-CutEdges]
    
    return Signal
 
def PlotWaveformsFromAFile(FName):
    #Plot all waveforms from a given file
    if (LoadFile(FName)==False):ErrorExit("DecodeChannels()")

    Waveforms,SumWaveforms = DecodeChannels(FName)
   
    for ch in range(NCh): 
        for i in range(len(Waveforms[ch])):
            plt.figure()
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (mV)")
            plt.title("Ch "+str(ch))
            plt.plot(Waveforms[ch][i],alpha=0.1)
       
    if NCh>1: 
        plt.figure()
        plt.xlabel("Time (a.u.)")
        plt.ylabel("Voltage (mV)")
        plt.title("Summed Waveforms")
        for i in range(len(ChSum)):
            plt.plot(ChSum[i],alpha=0.1)
                

def FileList(FPath):
    #For a given folder path, return the files
    FilePaths = str(FPath)+'/*.npy' #file paths of every .npy file
    FileList=glob.glob(FilePaths)
    size = os.path.getsize(FileList[len(FileList)-1])
    print(size)
    if(size==0):FileList = FileList[:-1] 
    return FileList

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
    ChSum = np.copy(ChDecoded[0])
    for i in range(1,NCh):
        ChSum+=ChDecoded[i]
    
    return ChDecoded, ChSum
    
def ProcessAWaveform(Ch,Signal,Filter,FreqF):
    #Extract information from a signal waveform:
    #    (Peak index, Peak value, Integrated charge, Noise RMS, Noise flag
    #Filter - If 1, apply a moving average filter
    #FreqF - If 1, applies a FFT to remove high frequency components
    
    if(Filter==1): Signal = MovAvFilter(Signal)
    if(FreqF ==1): Signal = FFT(Signal)
    ChT = np.linspace(0,len(Signal)*0.8,len(Signal)) #All channels have a dt = 0.8 ns
    
    #Calculate RMS of baseline area before signal window
    RMS = np.std(Signal[:SigLowerRMS])
    
    #Extract output analysis parameter from waveform
    PeakVal   = -np.min(Signal[SigLower:SigUpper])
    PeakIndex = np.argmin(Signal[SigLower:SigUpper])+SigLower
    ChargeVal = simps(Signal,ChT) # scipy integration function
    
    #Append outputs
    return WfInfo(Ch,PeakIndex,PeakVal,ChargeVal,RMS)
   
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

    return CurrentN , CurrentBins    

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
            wfInfo[ch] = ProcessAWaveform(ch,Waveforms[ch][i],1,1)
            if (wfInfo[ch].rms>RMS_Cut): NoisyEvent=True

        if (RemoveNoisyEvent==True and NoisyEvent==True): continue
        for ch in range(NCh):
            ChOutputs[ch].append(wfInfo[ch])
        ChSumOut.append(ProcessAWaveform('Sum',SumWaveforms[i],1,1))
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
        nBins, vals = PlotHistogram(heightArray,RangeLower,RangeUpper,NBins,str(dataArray.getChannel(0)),"Peak height [mV]")
        ChHistData.append(nBins)
        ChHistData.append(vals)
        
    #Plot summed channel histogram
    dataArray = ExtractWfInfo(SumOutputs)
    print(dataArray)
    heightArray = np.array(dataArray.getHeightArray(),dtype=np.float)
    nBins, vals = PlotHistogram(heightArray,4*RangeLower,4*RangeUpper,int(2.5*NBins),str(dataArray.getChannel(0)),"Peak height [mV]") 
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
    #FolderPath=r'C:\Users\smdek2\MPPCTests2021\Scint_Test_Oct15\Cosmic\\' 
    #FolderPath=r'/home/comet/work/pico/Oct15Cosmic//' 
    #nch,CosmicTR,CosmicHistData = AnalyseFolder(FolderPath,True)
    #CosmicBins = CosmicHistData[8]
    #CosmicN = CosmicHistData[9]
    
    #Analyse strontium data set
    #FolderPath=r'C:\Users\smdek2\MPPCTests2021\Scint_Test_Oct15\Strontium\\' 
    FolderPath=r'/home/comet/work/data/Dec13_TrigScintLargeScint_Sr90_SelfTrig50mV_Vb42_MinDist_2' 
    FolderPath=r'/home/comet/work/data/Dec10_LargeScint_Sr90ColEdge_SelfTrig10mV_Vb42//'
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
