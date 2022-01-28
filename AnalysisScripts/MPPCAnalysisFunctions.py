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
from scipy.stats import moyal

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
    return Signal - base

def FullAnalysis(Signal): # use this as a function to pass to PlotWaveformsFromAFile
    Signal = FFT(Signal)  # not actually directly used in this file
    Signal = MovAvFilter(Signal)
    Signal = BaselineFilter(Signal)
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
    #Filter - If True, apply a moving average filter
    #FreqF - If True, applies a FFT to remove high frequency components
    #Base - If True, estimates a low frequency voltage and subtracts it

    if Filter: Signal = MovAvFilter(Signal)
    if FreqF:  Signal = FFT(Signal)
    if Base:   Signal = BaselineFilter(Signal)
    ChT = np.linspace(0,len(Signal)*0.8,len(Signal)) #All channels have a dt = 0.8 ns

    #Calculate RMS of baseline area before signal window
    RMS = np.std(Signal[:SigLower])

    #Extract output analysis parameter from waveform
    PeakVal   = -np.min(Signal)
    PeakIndex = np.argmin(Signal)
    ChargeVal = simps(Signal,ChT) # scipy integration function

    #Append outputs
    return WfInfo(Ch,PeakIndex,PeakVal,ChargeVal,RMS)

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
    CurrentN,CurrentBins,_=plt.hist(Data,bins=NBins,color=colour,alpha=alpha)#range=[RL,RU])
    # I've removed range restrictions, but left the code commented if needed again
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
    if len(FList) == 0: # double checking files have been found as the error it would
        ErrorExit("AnalyseFolder: No files found") # give otherwise was confusing
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

    #Plot summed channel histogram
    dataArray = ExtractWfInfo(SumOutputs)
    print(dataArray)
    heightArray = np.array(dataArray.getHeightArray(),dtype=np.float)
    ChHeightData.append(heightArray)

    # Return data in form NCh, MeanTR, [ChABin,ChAN,...]
    ChHistData = ChannelHistograms(ChHeightData)
    # Histograms have been moved to ChannelHistograms for use in other Fns
    if RawHeightArray: return NCh, MeanTR, ChHistData, np.array(ChHeightData)
    # outputs the processed signal data as is for normalised PE fitting too
    return NCh, MeanTR, ChHistData

def ChannelHistograms(ChHeightArrays):
    ChHistData = []

    for ch in range(len(ChHeightArrays) - 1): # creates hists for each channel
        vals,nBins = PlotHistogram(ChHeightArrays[ch],RU,RL,NBins,str(ch),"Peak height [mV]")
        ChHistData.append(nBins)
        ChHistData.append(vals)

    vals,nBins = PlotHistogram(ChHeightArrays[-1],4*RU,4*RL,int(2.5*NBins),'Sum',"Peak height [mV]")
    ChHistData.append(nBins) # last heightarray is always assumed to be for sum signals
    ChHistData.append(vals)
    return [np.array(ChHistData[i]) for i in range(len(ChHistData))] # avoids deprecation warning

def MultiGaussianFit(B,N,String,properties): # applies fit of overlayed gaussians
    # properties = [spacing,theshold,Number of PE Peaks]
    spacing, threshold, NPE = properties
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
    return popt[1],p_sigma[1]

def LandauFit(B,N,String,properties): # ignores noise pedestal and applies moyal fit
    # args of form [peak,scale,minimum x value to avoid noise pedestal]
    peak1,scale1,threshold = properties
    ind1 = np.abs(B - peak1).argmin()
    NewB = B[B > threshold] # cuts off before threshold
    NewN = N[-len(NewB) + 1:]
    area = 1.2 * np.sum(NewN) * (NewB[1] - NewB[0])
    # estimates ~85% of moyal area is covered by the histogram data

    popt, pcov = curve_fit(ScaledMoyal,NewB[:-1],NewN,p0=[peak1,scale1,area])

    fit = ScaledMoyal(NewB,*list(popt)) # plottable data from the fit
    lw = 1

    plt.plot(NewB,fit,'k--',linewidth=lw)
    plt.title(String)
    plt.savefig('pics/Landau Fit ' + String + '.png', bbox_inches='tight')

    plt.figure() # plots log of N and the fit for better readability
    plt.title(String)
    plt.xlabel(XString)
    plt.ylabel('Log of Count')
    plt.bar(B[:-1],np.log10(N),width=B[1]-B[0], color='blue')
    plt.plot(NewB,np.log10(fit),'k--',lw=1)
    plt.ylim(top=1.1 * np.max(np.log10(N)),bottom=0)
    plt.savefig('pics/Landau Fit Log ' + String + '.png', bbox_inches='tight')

    p_sigma = np.sqrt(np.diag(pcov)) # print off moyal properties
    npopt, np_sigma = list(np.round(popt,2)), list(np.round(p_sigma,2))
    print('\nMoyal fit with peak at {} mV, scale {} and total area {}'.format(*npopt))
    print('Uncertainties are {}, {} and {} respectively'.format(*np_sigma))
    return popt,p_sigma

def ScaledMoyal(x,*params): # just a moyal fit with an area scaling factor too
    return params[2] * moyal.pdf(x,loc=params[0],scale=params[1])

def PE_Fitting(B,N,FitFn,args,String):  #outdated name as Landau fit doesn't get PE peaks
    #Input bins and counts and return single photoelectron analysis
    #B = bins of histogram
    #N = counts of histogram
    #fitfn = Gaussian or Landau fit function
    #args = guesses for the properties of the fitfn
    #String = string to title/label plot

    #Replot histogram using bar plot
    fig = plt.figure()
    plt.bar(B[:-1],N,width=B[1]-B[0], color='blue')
    plt.title(String)
    plt.xlabel(XString)
    plt.ylabel("Count")
    plt.savefig('pics/No Fit ' + String + '.png', bbox_inches='tight')
    try:
        return FitFn(B,N,String,args) # runs arbitrary fitting function
    except:
        print('Failed fitting\n')
        return 0 # NPE will not run if any multi gaussian fits fail

def NPE_Fitting(SinglePEs,FitFn,args,String,ChHeightData):
    #Same as PE Fitting except uses each channel's PE peak value to scale
    #each dataset and then sum them, only meant for the sum channel fitting
    #B = bins of histogram
    #N = counts of histogram
    #fitfn = Gaussian or Landau fit function
    #args = guesses for the properties of the fitfn
    #String = string to title/label plot

    SinglePEs = np.array(SinglePEs)
    PERatio = SinglePEs[:,0] / SinglePEs[0,0]
    ScaledData = ChHeightData[:-1] * PERatio[:,np.newaxis]
    SumData = np.sum(ScaledData,axis=0)
    SumHistData = ChannelHistograms([SumData])
    B,N = SumHistData[0],SumHistData[1]

    #Replot histogram using bar plot
    fig = plt.figure()
    plt.bar(B[:-1],N,width=B[1]-B[0], color='blue')
    plt.title(String)
    plt.xlabel(XString)
    plt.ylabel("Count")
    plt.savefig('pics/No Fit ' + String + '.png', bbox_inches='tight')
    return FitFn(B,N,String,args) # runs arbitrary fitting function

#Specific Analysis Functions - an analysis function for each data set!
#### Leave this for an example
def CosmicSr90Analysis(run='',PE=True,Sr=True,Cosmic=False):
    #Function to determine Sr90 spectrum from datasets of cosmic rays and Sr90 + cosmic rays

    #Analyse cosmic ray data set - note this is not a purely min ionising cosmic data set
    RawHeightArray = True
    DataPath = r'C:\Users\BemusedPopsicle\Desktop\Uni Stuff\ScinData\\' # folder with all data
    FolderPath = DataPath + r'2021-12-*-Near-Jacketed-' # specific folder(s) being run

    if run: run += ' '
    if Sr and not Cosmic: run += 'Sr90 '
    if Cosmic and not Sr: run += 'Cosmic ' # all just for formatting filenames and plots

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
    if Sr and Cosmic: PEHeightData,PEHistData=CosmicHeightData,CosmicHistData #manual choice
    if Sr and not Cosmic: PEHeightData,PEHistData = SrHeightData,SrHistData
    if not Sr and Cosmic: PEHeightData,PEHistData = CosmicHeightData,CosmicHistData
    if PE: # apply fitting to channel and sum histograms
        spacing = 3 # estimated distance between peaks (MultiGaussian)
        threshold = 100 # minimum counts to be a peak (MultiGaussian)
        SinglePEs = []
        FitFn = LandauFit # function for sum analysis
        MGargs = [spacing,threshold,NPks] # args for channel fitting
        if FitFn == MultiGaussianFit: args = [spacing,threshold,NPks] # args for sum fitting
        if FitFn == LandauFit and 'Unjacketed' in run: args = [180,30,80]
        if FitFn == LandauFit and 'Jacketed' in run: args = [70,20,29]

        for i in range(NCh):
            title = run + 'Channel ' + chr(ord('A') + i) + ' PE Peaks'
            SinglePEs.append(PE_Fitting(PEHistData[2*i],PEHistData[2*i+1],MultiGaussianFit,MGargs,title))

        title = run + 'Channel Sum'
        if 0 in SinglePEs: # make sure it only uses NPE if it can scale each dataset properly
            PE_Fitting(PEHistData[8],PEHistData[9],FitFn,args,title)
        else: NPE_Fitting(SinglePEs,FitFn,args,title,PEHeightData)

    if not (Sr and Cosmic): return
    plt.figure() # plots spectra if Sr and Cosmic datasets have been run together
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
