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
SigLower = 80
SigUpper = 110

NBins = 100 #Histogram bins used
RU    = 35 #Upper limit of histograms
RL    = 0 #Lower limit of histograms
XString = 'Peak Voltage [mV]'
NPks = 7 # maximum number of peaks to fit to

#Set a baseline RMS cut off to remove noise
RMS_Cut = 3.0 #mV (based on plotting RMS values for baseline window [:50])

##def Initialise(): # to be implemented

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

def BaselineFilter(Signal): #n is pretty arbitrary, just where peaks don't pop up yet
    n = 150 #number of data points from start to consider
    base = np.mean(Signal[:n])
    return Signal - base #this is also a somewhat crude implementation

def FullAnalysis(Signal):
    Signal = FFT(Signal)
    Signal = MovAvFilter(Signal)
    Signal = BaselineFilter(Signal)
    return Signal

def NoBaseline(Signal):
    Signal = FFT(Signal)
    Signal = MovAvFilter(Signal)
    return Signal

def PlotWaveformsFromAFile(FName,fn=None,SingleWf=0,SplitCh=False,title=''):
    #Plot all waveforms from a given file
    if (LoadFile(FName)==False):ErrorExit("DecodeChannels()")

    Waveforms,SumWaveforms = DecodeChannels(FName)

    if fn != None:
        NewWfs = [[],[],[],[]]
        NewSumWfs = []
        for i in range(len(SumWaveforms)):
            for ch in range(NCh):
                NewWfs[ch].append(fn(Waveforms[ch][i]))
            NewSumWfs.append(fn(SumWaveforms[i]))
        Waveforms = np.array(NewWfs)
        SumWaveforms = np.array(NewSumWfs)

    for i in range(SingleWf):
        if not SplitCh: plt.figure()
        for ch in range(NCh):
            if SplitCh:
                plt.figure()
                plt.title(title + "Ch "+str(ch))
            else:
                plt.title(title + "All Channels")
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (mV)")
            plt.plot(Waveforms[ch][i],alpha=0.1)

    if NCh>1:
        plt.figure()
        plt.xlabel("Time (a.u.)")
        plt.ylabel("Voltage (mV)")
        plt.title(title + "Summed Waveforms")
        for i in range(len(SumWaveforms)):
            plt.plot(SumWaveforms[i],alpha=0.1)


def FileList(FPath):
    #For a given folder path, return the files
    FilePaths = str(FPath)+'/*.npy' #file paths of every .npy file
    FileList=glob.glob(FilePaths)

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

def ProcessAWaveform(Ch,Signal,Filter,FreqF,Base):
    #Extract information from a signal waveform:
    #    (Peak index, Peak value, Integrated charge, Noise RMS, Noise flag
    #Filter - If 1, apply a moving average filter
    #FreqF - If 1, applies a FFT to remove high frequency components

    if(Filter==1): Signal = MovAvFilter(Signal)
    if(FreqF ==1): Signal = FFT(Signal)
    if(Base  ==1): Signal = BaselineFilter(Signal)
    ChT = np.linspace(0,len(Signal)*0.8,len(Signal)) #All channels have a dt = 0.8 ns

    #Calculate RMS of baseline area before signal window
    RMS = np.std(Signal[:SigLower])

    #Extract output analysis parameter from waveform
    PeakVal   = -np.min(Signal)
    PeakIndex = np.argmin(Signal)
    ChargeVal = simps(Signal,ChT) # scipy integration function

    #Append outputs
    return WfInfo(Ch,PeakIndex,PeakVal,ChargeVal,RMS)

def HistogramFitOld(x,*params):
    y = np.zeros_like(x)
    for i in range(0,len(params),3):
        mean = params[i]
        amplitude = params[i+1]
        sigma = params[i+2]
        y = y + amplitude * np.exp(-((x-mean)/sigma)**2)
    return y

def HistogramFit(x,*params):
    #n = number of peaks
    n = NPks
    y = np.zeros_like(x)
    Width = params[1]
    PedMean=params[0]
    for i in range(n):
        #mean = params[0]+params[i]
        mean = PedMean+i*Width
        amplitude = params[2*i+2]
        sigma = params[2*i+3]
        y = y + amplitude * np.exp(-((x-mean)/sigma)**2)
    return y

def PlotHistogram(Data,RU,RL,NBins,String,strData): #pdist,threshold,subplot,Ped_Peak,SP_Peak,uPed,uSP):
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
            wfInfo[ch] = ProcessAWaveform(ch,Waveforms[ch][i],1,1,1)
            if (wfInfo[ch].rms>RMS_Cut): NoisyEvent=True

        if (RemoveNoisyEvent==True and NoisyEvent==True): continue
        for ch in range(NCh):
            ChOutputs[ch].append(wfInfo[ch])
        ChSumOut.append(ProcessAWaveform('Sum',SumWaveforms[i],1,1,1))
    #print(ChSumOut)
    ## Prepare to read the next file
    FileLoaded = False
    return TRate

def AnalyseFolder(FPath,RawHeightArray=False):
    #Analyse all data files in folder located at FPath
    #RawHeightArray is an option to output the full height arrays

    MeanTR = 0

    FList = FileList(FPath)
    TriggerRates = []

    FileOutputs = [[],[],[],[]]  # 4channels
    SumOutputs = []
    for i in range(len(FList)):
        print("Analysing file:",FList[i][len(FPath)-1:])
        TRate = AnalyseSingleFile(FList[i],FileOutputs,SumOutputs)
        TriggerRates.append(TRate)
        print("Trigger rate (Hz) = ",TRate)

    TriggerRates=np.array(TriggerRates)
    MeanTR = np.mean(TriggerRates)

    ChHistData = []
    ChHeightData = []
    for ch in range(NCh):
        dataArray = ExtractWfInfo(FileOutputs[ch])
        print(dataArray)
        heightArray = np.array(dataArray.getHeightArray(),dtype=np.float)
        ChHeightData.append(heightArray)
        vals, nBins = PlotHistogram(heightArray,RU,RL,NBins,str(dataArray.getChannel(0)),"Peak height [mV]")
        ChHistData.append(nBins)
        ChHistData.append(vals)

    #Plot summed channel histogram
    dataArray = ExtractWfInfo(SumOutputs)
    print(dataArray)
    heightArray = np.array(dataArray.getHeightArray(),dtype=np.float)
    ChHeightData.append(heightArray)
    vals, nBins = PlotHistogram(heightArray,4*RU,4*RL,int(2.5*NBins),str(dataArray.getChannel(0)),"Peak height [mV]")
    ChHistData.append(nBins)
    ChHistData.append(vals)

    # Return data in form NCh, MeanTR, [ChABin,ChAN,...]
    ChHistData = [np.array(ChHistData[i]) for i in range(len(ChHistData))]
    if RawHeightArray: return NCh, MeanTR, ChHistData, ChHeightData
    return NCh, MeanTR, ChHistData

def PE_Fitting(B,N,spacing,threshold,NPE,String):
    #Input bins and counts and return single photoelectron analysis
    #B = bins of histogram
    #N = counts of histogram
    #spacing = estimated spacing between peaks (in bins)
    #threshold = minimum peak height (in counts)
    #NPE = number of peaks in histogram to fit to (may struggle with more than two)
    #String = string to title/label plot

    #Replot histogram using bar plot
    fig = plt.figure()
    plt.bar(B[:-1],N,width=B[1]-B[0], color='blue')
    plt.title(String)
    plt.xlabel(XString)
    plt.ylabel("Count")
    plt.savefig('pics/No Fit ' + String + '.png', bbox_inches='tight')
    peaks, _ = find_peaks(N, distance=spacing,height=threshold)
    NPE = np.min((NPE,len(peaks)))
    global NPks
    NPks = NPE


    #Use scipy find_peaks as guesses for fit (if the peaks aren't looking right you can play with the spacing and threshold variables)
    PeakX = B[peaks] #peak voltage of hist peaks
    PeakY = N[peaks] #amplitudes of hist peaks
    PeakI = peaks #indices of hist peaks

    #Add peak markers to plot
    plt.scatter(PeakX,PeakY,s=20,marker='x',color='k')

    if(PeakX.size<2):
        ErrorExit("Pedestal fitting (find_peaks)")

    #Guess of fitting window
    PEWidth = PeakX[1]-PeakX[0]
    PEWidthI = PeakI[1]-PeakI[0]
    FitCutGuessI = int(PeakI[0]+round((NPE+0.5)*PEWidthI))

    WindowAdjustment = 2
    XWindow = B[:FitCutGuessI+WindowAdjustment]
    YWindow = N[:FitCutGuessI+WindowAdjustment]

    #Fitting Guess in form params = [mean,amplitude, sigma] for each gaussian
    guess_pedestal = PeakX[0]

    guess_params = [guess_pedestal,PEWidth,PeakY[0],PEWidth/2]
    for i in range(NPE):
        #Initial peak height guess is that it approximately halves for each
        guess_params += [PeakY[1]/(2**i),PEWidth/2]
    #print(guess_params,XWindow,YWindow)

    popt, pcov = curve_fit(HistogramFit,XWindow,YWindow,p0=guess_params)

    p_sigma = np.sqrt(np.diag(pcov))
    fit = HistogramFit(XWindow,*popt)
    lw = 1

    plt.plot(XWindow,fit,'k--',linewidth=lw)
    plt.title(String + ' (Single P.E. = %.2f)' %(popt[1]))
    plt.savefig('pics/Fit ' + String + '.png', bbox_inches='tight')

    #Print 1 p.e. value - NOTE: There is some bug with uncertainties in fit at this moment
    SinglePE=0
    uSinglePE=0
    print(String)
    # print(popt)
    # print(p_sigma)
    print("Single P.E. : V = %.2f +- %.2f\n" %(popt[1],p_sigma[1]))
    return

#Specific Analysis Functions - an analysis function for each data set!
#### Leave this for an example
def CosmicSr90Analysis(run='',PE=True,Sr=True,Cosmic=False):
    #Function to determine Sr90 spectrum from datasets of cosmic rays and Sr90 + cosmic rays

    #Analyse cosmic ray data set - note this is not a purely min ionising cosmic data set
    RawHeightArray = True
    DataPath = r'C:\Users\BemusedPopsicle\Desktop\Uni Stuff\ScinData'
    FolderPath = DataPath + r'\2021-12-*-Middle-'

    if run: run += ' '

    if Cosmic:
        Folder = FolderPath + r'Cosmic\\'
        if RawHeightArray: nch,CosmicTR,CosmicHistData,CosmicHeightData = AnalyseFolder(Folder,RawHeightArray)
        else: nch,CosmicTR,CosmicHistData = AnalyseFolder(Folder,RawHeightArray)
        CosmicBins = CosmicHistData[8]
        CosmicN = CosmicHistData[9]

    #Analyse strontium data set
    if Sr:
        Folder = FolderPath + r'Strontium\\'
        if RawHeightArray: nch,SrTR,SrHistData,SrHeightData = AnalyseFolder(Folder,RawHeightArray)
        else: nch,SrTR,SrHistData = AnalyseFolder(Folder,RawHeightArray)
        SrBins = SrHistData[8]
        SrN = SrHistData[9]

    #Subtract cosmic spectrum from strontium + cosmic spectrum
    if Sr and Cosmic:
        nCosmicN = CosmicTR*CosmicN/np.sum(CosmicN)
        print("Mean cosmic trigger rate = ", CosmicTR)
        nSrN = SrTR*SrN/np.sum(SrN)
        print("Mean cosmic + Sr90 trigger rate = ", SrTR)
        SrSpectrum = nSrN-nCosmicN

    #Plot histogram using bar plot
    PEHistData = CosmicHistData
    PEHeightData = CosmicHeightData
    if PE:
        spacing = 3 # estimated distance between peaks
        threshold = 100 # minimum counts to be a peak
        for i in range(NCh):
            title = run + 'Sr90 Channel ' + chr(ord('A') + i) + ' PE Peaks'
            PE_Fitting(PEHistData[2*i],PEHistData[2*i+1],spacing,threshold,NPks,title)#,PEHeightData)
        # PE_Fitting(PEHistData[8],PEHistData[9],spacing,threshold,NPks,'Sr90 Channel Sum PE Peaks')
    if not (Sr or Cosmic): return
    plt.figure()
    plt.bar(CosmicBins[:-1],nCosmicN,width=CosmicBins[1]-CosmicBins[0], color='blue')
    plt.title(run+"Strontium and Cosmic Spectrum")
    plt.ylabel("TR * Count")

    plt.figure()
    plt.bar(SrBins[:-1],nSrN,width=SrBins[1]-SrBins[0], color='blue')
    plt.title(run+"Cosmic Spectrum")
    plt.ylabel("TR * Count")

    plt.figure()
    plt.bar(SrBins[:-1],SrSpectrum,width=SrBins[1]-SrBins[0], color='blue')
    plt.title(run+"Reverse Strontium Spectrum")
    plt.xlabel(XString)
    plt.ylabel("Sr TR * Count - Cos TR * Count")

    #Determine endpoint of spectrum?
    return

#############Main - run analysis functions here
#FolderPath=r'C:\Users\smdek2\MPPCTests2021\Scint_Test_Oct15\Strontium\\'
#PlotRMS(FolderPath,"Sr90")
#PlotWaveformsFromAFile(FolderPath+"ScintTestOct15_Sr90_T9mV_99.npy")
#CosmicSr90Analysis()
#plt.show()
