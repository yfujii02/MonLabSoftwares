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

FileLoaded = False
FileData   = []
HeaderInfo = []
ChData     = []
RemoveNoisyEvents=True
NCh = 0

#Global variables
SigLower = 80
SigUpper = 110

NBins = 50 #Histogram bins used 
RU  = 50 #Upper limit of histograms 
RL = 5 #Lower limit of histograms 

#Global RMS data arrays
NoiseRMSA=[]
NoiseRMSB=[]
NoiseRMSC=[]
NoiseRMSD=[]

#Set a baseline RMS cut off to remove noise
RMS_Cut = 3.0 #mV (based on plotting RMS values for baseline window [:50])

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
 
def PlotWaveforms(iData,Data,PlotList,String):
    #Plot all waveforms from a given file
    #PlotList - if 0, plot all, otherwise treat as list to plot
    #String - heading string
    plt.figure()
    plt.xlabel("Time (ns)")
    plt.ylabel("Voltage (mV)")
    plt.title(String)
    if(PlotList==0):
        for i in range(len(Data)):
            plt.plot(iData,Data[i],alpha=0.1)
    else:
        for i in range(len(PlotList)):
            plt.plot(iData,Data[PlotList[i]],alpha=0.1)
    return

def PlotFile(FName):
    #Plotlist is a list of indices to plot, if 0 plot all
    plt.figure()
    plt.xlabel("Time (ns)")
    plt.ylabel("Voltage (mV)")
    plt.title("Ch1")
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
    FilePaths = str(FPath)+'*.npy' #file paths of every .npy file
    FileList=glob.glob(FilePaths)
    
    return FileList

def LoadFile(FName):
    #Returns number of channels
    global FileData
    global HeaderInfo
    global NCh
    global ChData
    global FileLoaded
    FileData   = np.array(np.load(FName,allow_pickle=True))
    HeaderInfo = np.array([FileData[:6]])
    ChData     = FileData[6]
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
    for i in range(int(len(Data[:,0])/NCh)):
        ## Loop over the different channels
        for j in range(NCh):
            ChDecoded[j].append(Data[NCh*i+j,:])
    
    ## Make a summed waveform
    for i in range(NCh):
        ChDecoded[i] = np.array(ChDecoded[i])
    ChSum = np.copy(ChDecoded[0])
    for i in range(1,NCh):
        ChSum+=ChDecoded[i]
    
    return ChDecoded, ChSum
    
def ProcessAWaveform(Signal,Filter,FreqF):
    #Extract information from a signal waveform:
    #    (Peak index, Peak value, Integrated charge, Noise RMS, Noise flag
    #Filter - If 1, apply a moving average filter
    #FreqF - If 1, applies a FFT to remove high frequency components
    WaveformInfo = []
    
    if(Filter==1): Signal = MovAvFilter(Signal)
    if(FreqF ==1): Signal = FFT(Signal)
    ChT = np.linspace(0,len(Signal)*0.8,len(Signal)) #All channels have a dt = 0.8 ns
    
    #Calculate RMS of baseline area before signal window
    RMS = np.std(Signal[:SigLower])
    
    #Extract output analysis parameter from waveform
    PeakVal   = -np.min(Signal)
    PeakIndex = np.argmin(Signal)
    ChargeVal = simps(Signal,ChT) # scipy integration function
    
    #Append outputs
    WaveformInfo.append(PeakIndex)
    WaveformInfo.append(PeakVal)
    WaveformInfo.append(ChargeVal)
    WaveformInfo.append(RMS)
    return np.array(WaveformInfo)
   
def HistogramFit(x,*params):
    y = np.zeros_like(x)
    for i in range(0,len(params),3):
        mean = params[i]
        amplitude = params[i+1]
        sigma = params[i+2]
        y = y + amplitude * np.exp(-((x-mean)/sigma)**2)
    return y  
  
def PlotHistogram(Data,RU,RL,NBins,String): #pdist,threshold,subplot,Ped_Peak,SP_Peak,uPed,uSP):
    #Take collected channel data from all files to be analysed and plot histogram
    #Data for a given channel
    #RU,RL = range of histogram
    #NBins = number of bins
    #String = title string on plot
    
    colour = 'purple'
    alpha = 0.5
    
    plt.figure()
    CurrentN,CurrentBins,_=plt.hist(Data,range=[RL,RU],bins=NBins,color=colour,alpha=alpha)
    plt.title(String)
    plt.xlabel(XString)
    plt.ylabel("Count")

    return CurrentN , CurrentBins    

def AnalyseSingleFile(FName,ChOutputs,ChSumOut):
    #Takes a file path and analyses all waveforms in the file
    #Returns an output with form [[PeakIndex, PeakValue, ChargeValue, BaselineRMS],...,[]] containing info for each waveform
        #PeakIndex   = Index within file of analysed value (i.e. index of signal peak)
        #PeakValue   = Analysed value (i.e. peak value or integrated charge)
        #ChargeValue = Analysed value (i.e. peak value or integrated charge)
        #BaselineRMS = RMS (standard deviation) of Baseline
    #This is returned for each channel separately
    if (FileLoaded==False): LoadFile(FName)
    
    ChDecoded,ChSum = DecodeChannels(FName)
    
    NWaveforms = len(ChDecoded[0]) #All channels have same number of waveforms
    TRate = (HeaderInfo[0][4]/(HeaderInfo[0][3]-HeaderInfo[0][2]))
    
    for i in range(NWaveforms):
        NoisyEvent=False
        WfInfo=[[],[],[],[]]
        #### check each channel
        for ch in range(NCh):
            WfInfo[ch] = ProcessAWaveform(ChDecoded[i],1,1)
            if (WfInfo[ch][3]>RMS_Cut): NoisyEvent=True

        if (RemoveNoisyEvent==True and NoisyEvent===True): continue
        for ch in range(NCh):
            ChOutputs[ch].append(WfInfo[ch])
        ChSumOut.append(ProcessAWaveform(ChSum[i],1,1))
    #print(ChSumOut)
    return TRate
    
def PlotRMS(FolderPath,String):
    #Plot histogram of baseline RMS for an entire folder of data
    #Use string to append to title of figures
    Flist = FileList(FolderPath)
    
    #Set and reset data arrays
    global NoiseRMSA
    global NoiseRMSB
    global NoiseRMSC
    global NoiseRMSD
    
    NoiseRMSA=[]
    if(NCh>1): NoiseRMSB=[]
    if(NCh>2): NoiseRMSC=[]
    if(NCh>3): NoiseRMSD=[]
    
    #Analyse folder
    AnalyseFolder(FolderPath,0)
    
    #Plot individual channels
    PlotHistogram(NoiseRMSA,10,0,50,String+" Baseline RMS Ch A")
    if(NCh>1): PlotHistogram(NoiseRMSB,10,0,50,String+" Baseline RMS Ch B")
    if(NCh>2): PlotHistogram(NoiseRMSC,10,0,50,String+" Baseline RMS Ch C")
    if(NCh>3): PlotHistogram(NoiseRMSD,10,0,50,String+" Baseline RMS Ch D")
    
    return


def AnalyseFolder(FPath,PlotFlag):
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
    
    ChA_Vals = CollectOutputs(ChAOut,1)
    #SummedHist = ChA_Vals
    
    ChA_N, ChA_Bin = PlotHistogram(ChA_Vals,RU,RL,NBins,"Ch A")
    
    ChHistData.append(ChA_Bin)
    ChHistData.append(ChA_N)
    
    if(NCh>1):
        ChB_Vals =CollectOutputs(ChBOut,1)
        #SummedHist += ChB_Vals
        ChB_N, ChB_Bin = PlotHistogram(ChB_Vals,RU,RL,NBins,"Ch B")
        ChHistData.append(ChB_Bin)
        ChHistData.append(ChB_N)
        
    if(NCh>2):
        ChC_Vals =CollectOutputs(ChCOut,1)
        #SummedHist += ChC_Vals
        ChC_N, ChC_Bin = PlotHistogram(ChC_Vals,RU,RL,NBins,"Ch C")
        ChHistData.append(ChC_Bin)
        ChHistData.append(ChC_N)
        
    if(NCh>3):
        ChD_Vals =CollectOutputs(ChDOut,1)
        #SummedHist += ChD_Vals
        ChD_N, ChD_Bin = PlotHistogram(ChD_Vals,RU,RL,NBins,"Ch D")
        ChHistData.append(ChD_Bin)
        ChHistData.append(ChD_N)
        
    #Plot summed channel histogram
    ChSum_Vals = CollectOutputs(ChSumOut,1)
    Total_N, Total_Bin = PlotHistogram(ChSum_Vals,4*RU,4*RL,int(2.5*NBins),"Total") 
    ChHistData.append(Total_Bin)
    ChHistData.append(Total_N)
    
    # Return data in form NCh, MeanTR, [ChABin,ChAN,...]
    ChHistData = np.array(ChHistData)
    
    return NCh, MeanTR, ChHistData 


#Specific Analysis Functions - an analysis function for each data set!
def CosmicSr90Analysis():
    #Function to determine Sr90 spectrum from datasets of cosmic rays and Sr90 + cosmic rays
    
    #Analyse cosmic ray data set - note this is not a purely min ionising cosmic data set
    #FolderPath=r'C:\Users\smdek2\MPPCTests2021\Scint_Test_Oct15\Cosmic\\' 
    FolderPath=r'/home/comet/work/pico/Oct15Cosmic//' 
    nch,CosmicTR,CosmicHistData = AnalyseFolder(FolderPath,1)
    CosmicBins = CosmicHistData[8]
    CosmicN = CosmicHistData[9]
    
    #Analyse strontium data set
    #FolderPath=r'C:\Users\smdek2\MPPCTests2021\Scint_Test_Oct15\Strontium\\' 
    FolderPath=r'/home/comet/work/pico/ReverseStrontium//' 
    nch,SrTR,SrHistData = AnalyseFolder(FolderPath,1)
    SrBins = SrHistData[8]
    SrN = SrHistData[9]
    
    #Subtract cosmic spectrum from strontium + cosmic spectrum
    nCosmicN = CosmicTR*CosmicN/np.sum(CosmicN)
    print("Mean cosmic trigger rate = ", CosmicTR)
    nSrN = SrTR*SrN/np.sum(SrN)
    print("Mean cosmic + Sr90 trigger rate = ", SrTR)
    SrSpectrum = nSrN-nCosmicN    

    #Plot histogram using bar plot
    plt.figure()
    plt.bar(SrBins[:-1],SrSpectrum,width=SrBins[1]-SrBins[0], color='blue') 
    plt.title("Reverse Strontium Spectrum")
    plt.xlabel(XString)
    plt.ylabel("Count")
    
    #Determine endpoint of spectrum?
    
    return

def SetSignalWindow(sigL,sigU):
    global SigLower
    global SigUpper
    
    SigLower = sigL
    SigUpper = sigU

def MinIonisationCosmicData():
    
    #FolderPath=r'C:\Users\smdek2\MPPCTests2021\Scint_Test_Oct15\Cosmic\\' 
    FolderPath = r'/home/comet/work/pico/Oct21//'
    nch,TR,HistData = AnalyseFolder(FolderPath,1)
    SumBins = HistData[8]
    SumN = HistData[9]
    
    #Plot histogram using bar plot
    plt.figure()
    plt.bar(SumBins[:-1],SumN,width=SumBins[1]-SumBins[0], color='blue') 
    plt.title("Strontium Spectrum")
    plt.yscale('log')
    plt.xlabel(XString)
    plt.ylabel("Count")
    
    return

def Pb210():    
    #Analyse PB210 data   
 
    #Analyse cosmic ray data set - note this is not a purely min ionising cosmic data set
    FolderPath=r'/home/comet/work/pico/Oct15Cosmic//' 
    nch,CosmicTR,CosmicHistData = AnalyseFolder(FolderPath,1)
    CosmicBins = CosmicHistData[8]
    CosmicN = CosmicHistData[9]
    
    #Analyse lead data set
    FolderPath = r'/home/comet/work/pico/Oct22//'
    nch,PbTR,PbHistData = AnalyseFolder(FolderPath,1)
    PbBins = PbHistData[8]
    PbN = PbHistData[9]
    
    #Subtract cosmic spectrum from lead + cosmic spectrum
    nCosmicN = CosmicTR*CosmicN/np.sum(CosmicN)
    print("Mean cosmic trigger rate = ", CosmicTR)
    nPbN = PbTR*PbN/np.sum(PbN)
    print("Mean cosmic + Pb210 trigger rate = ", PbTR)
    PbSpectrum = nPbN-nCosmicN    

    #Plot histogram using bar plot
    plt.figure()
    plt.bar(PbBins[:-1],PbSpectrum,width=PbBins[1]-PbBins[0], color='blue') 
    plt.title("Pb210 Spectrum")
    plt.xlabel(XString)
    plt.ylabel("Count")

    return    


#############Main - run analysis functions here


#FolderPath=r'C:\Users\smdek2\MPPCTests2021\Scint_Test_Oct15\Strontium\\' 
#PlotRMS(FolderPath,"Sr90")
#PlotFile(FolderPath+"ScintTestOct15_Sr90_T9mV_99.npy")

CosmicSr90Analysis()
plt.show()
