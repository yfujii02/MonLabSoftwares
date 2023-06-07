# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:19:53 2020

Analysis Function Package to provide the general waveform analysis
for the data taken by using picoscope and saved as npy file.
 
@author: smdek2
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.integrate import simps
from scipy import fftpack
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import glob 


#FFT Global Parameters
FFTOn=1
Samples = 396 
dt = 0.8e-9
TimeArray = np.linspace(0.0,Samples*dt,Samples)
fftT = np.linspace(0.0, 1.0/(2.0*dt), Samples//2)
ThresholdFreq = 1e4

IntegrationWidth=40
NumMAv=5 # Number of moving average points
PedestalRange=90

def SetFFTOn(ffton):
    global FFTOn
    FFTOn = ffton
    return 1

def GetFileList(DataPath,Date,FibreType,FibreLength,LEDWavelength,Voltage,RunNumber,Prefix):
    #Example: DataPath = r'C:\Users\smdek2\Documents\SiPMTests\PlasticFibreTests\Sep17_InitialGlassFibreTests\'
    #         Date = Sep17
    #         Voltage = 42
    #         FibreLength = 10  
    #         FibreType = glass
    #         LEDWavelength = 405
    #         Run = 2
    if(Prefix=='temp'):
        FilePaths = str(DataPath)+str(Date)+'_Overnight_'+str(RunNumber)+'*.npy'
    else: 
        if(RunNumber==0): FilePaths = str(DataPath)+str(Date)+'_hv'+str(Voltage)+'_'+str(FibreLength)+'m_'+str(FibreType)+'_LED'+str(LEDWavelength)+'nm_'+str(Prefix)+'*.npy'
        else: FilePaths = str(DataPath)+str(Date)+'_hv'+str(Voltage)+'_'+str(FibreLength)+'m_'+str(FibreType)+'_LED'+str(LEDWavelength)+'nm_'+str(Prefix)+str(RunNumber)+'*.npy'
    #print("File Path = ",FilePaths)
    FileList=glob.glob(FilePaths)    
    return FileList

#### Get waveforms from the npy file
def GetWaveformData(Filename):
    FileData = np.array(np.load(Filename,allow_pickle=True))
    HeaderData = np.array([FileData[:6]])            
    WaveformData = FileData[6]
    return WaveformData, HeaderData

#### Compatible to the original peak search function written by SamD
def PeakSearchOrig(wf,peakSearchRange,peakThreshold,baseLevel,maxIndex):
    startIndex=-1
    endIndex  =-1
    waveform  = wf[peakSearchRange[0]:peakSearchRange[1]]
    if waveform.max()<peakThreshold:
        ### Return if no peak found...
        return False,startIndex,endIndex
    maxIndex=np.argmax(waveform)
    for i in range(maxIndex,0,-1):
        if(waveform[i]<=baseLevel):
            startIndex=i
            break
    for i in range(maxIndex,len(waveform),1):
        if(waveform[i]<=baseLevel):
            endIndex=i
            break
    if (startIndex<0 or endIndex<0):
        return False,startIndex,endIndex
    startIndex = startIndex+peakSearchRange[0]
    endIndex   = endIndex  +peakSearchRange[0]
    return True,startIndex,endIndex

def PeakSearchSciPy(wf,heightThr,minGap):
    find_peaks(wf,height=heightThr,distance=minGap)
    return None

def GetData(WaveformData,Output,Length, DataType,SinglePhotonOutput):
    #Data Type: 0 = PeakVoltage, 1 = Integrated Charge
    #SinglePhotonOutput is for recording peak values in first 100 s
    CurrentIntegrated=[]
    CurrentMaxPeaks=[]
    CurrentStartIndices=[]
    CurrentStopIndices=[]
    LEDSigPlot=0
    BadEventCounter=0
    PeakPlot=0
    if(PeakPlot==1 or LEDSigPlot==1):
        fig = plt.figure()
        plt.ion()
    for j in range(int(len(WaveformData)/2)):
       
        #FilterWaveform
        MPPCSig = WaveformData[2*j,:]
        LEDSignal = WaveformData[2*j+1,:]
        
        #print("Waveform size = ",len(MPPCSig)) This is 400
        moveavefilt=np.cumsum(np.insert(MPPCSig,0,0))
        MPPCSig=(moveavefilt[NumMAv:]-moveavefilt[:-NumMAv])/float(NumMAv)

        #baseLevel=np.array(MPPCSig[NumMAv:NumMAv+PedestalRange]).mean()
        #baseStd  =np.array(MPPCSig[NumMAv:NumMAv+PedestalRange]).std()
        #PeakThreshold=10
        PeakThreshold = 90
        if(np.max(MPPCSig)<30): PeakThreshold=5
        
        if(Length==1): TimeThreshold=120
        if(Length==3): TimeThreshold=130
        else: TimeThreshold=150
        FFTOn=0
        #print("Here")
        #Apply FFT for easier main peak finding
        if(FFTOn==1):
            fftV = fftpack.fft(MPPCSig)   
            samplefreq=fftpack.fftfreq(Samples, dt)
            #plt.figure()
            #plt.plot(fftV)
            Copy = fftV.copy()
            Copy[np.abs(samplefreq)>ThresholdFreq]=0
            filteredsig=fftpack.ifft(Copy)
            MPPCSigFilt = filteredsig.real
        else:
            MPPCSigFilt = MPPCSig
        
        #Find peak
        MaxIndex=-1
        StartIndex=-1
        EndIndex=-1
        PeakFlag=-1
        PeakFound=False
        SearchRange=[TimeThreshold,len(MPPCSigFilt)-1] 
        #print(SearchRange)
        PeakFound,StartIndex,EndIndex=PeakSearchOrig(MPPCSigFilt,SearchRange,PeakThreshold,0,MaxIndex)
        
        
        
        if(PeakFound==False):
            BadEventCounter+=1
            #Pick an approximate fixed window to integrate over
            if(Length==1):
                StartIndex = 120
                EndIndex = StartIndex+IntegrationWidth
            else:
                StartIndex = 180
                EndIndex = StartIndex+IntegrationWidth
        
        if(PeakPlot==1):
            SigRange=np.linspace(StartIndex,EndIndex,(EndIndex-StartIndex)+1)
            
            plt.plot(MPPCSig, color='k')
            plt.plot(SigRange,MPPCSig[StartIndex:EndIndex+1], color='r')
            plt.plot(MPPCSigFilt,color='blue')
            plt.draw()
            plt.pause(0.01)
            fig.clear()
        if(LEDSigPlot==1):
            plt.plot(LEDSignal)
            plt.draw()
            plt.pause(0.01)
            fig.clear()
           
        CurrentStartIndices.append(StartIndex)
        CurrentStopIndices.append(EndIndex)
        
        #Numerical Integration using Simpsons Rule 
        if(StartIndex>=EndIndex): Area = 0.0
        else: Area = simps((MPPCSig)[StartIndex:EndIndex],TimeArray[StartIndex:EndIndex])
     
        CurrentIntegrated.append(Area*1e7)
        if(DataType==0): 
            Baseline=np.mean(MPPCSig[0:125])
            Output.append(np.max(MPPCSig)-Baseline)
        if(DataType==1): Output.append(Area*1e7)
        
        if(DataType==2):
            Output.append(np.max(LEDSignal))
            Mean1 = np.mean(LEDSignal[0:50])
            Mean2 = np.mean(LEDSignal[200:])
            Output.append((Mean1+Mean2)/2.0)
        #CurrentMaxPeaks.append(np.max(MPPCSig[StartIndex:EndIndex]))
        CurrentMaxPeaks.append(np.max(MPPCSig))
        
        #Search first 100 values for single photon peak
        FoundSinglePeak=0
        SinglePeakSearch = MPPCSig[0:100]
        #while(FoundSinglePeak==0):
            #SinglePeak = SinglePeakSearch.index(np.max(SinglePeakSearch))
        SinglePeakIndex = np.argmax(SinglePeakSearch)
        SinglePeak = SinglePeakSearch[SinglePeakIndex]
        #if(SinglePeak>20.0 or SinglePeak<8.0):
        #    SinglePeak=50
         #   else:
        #FoundSinglePeak=1
        #print(SinglePeak)
        SinglePhotonOutput.append(SinglePeak)

    return 1


def GetHistogram(Data,DataType,NBins,RL,RU,Guess,FibreLength,FibreType,Wavelength,SuppressPlots):
    #Data Type: 0 = PeakVoltage, 1 = Integrated Charge
    
    if(SuppressPlots==0):fig = plt.figure()
    alpha=0.5
    colour='red'    
    CurrentN,CurrentBins,_=plt.hist(Data,bins=NBins,density = True,range = (RL,RU),alpha=alpha,color=colour,label='Data')
    if(DataType==1): plt.title("PV: "+str(FibreLength)+" m "+str(FibreType)+" with "+str(Wavelength)+" nm LED") 
    plt.xlim(RL, RU)
    if(DataType==1): plt.xlabel("Integrated charge (1e-7)")
    
    if(DataType==0): plt.title("PV: "+str(FibreLength)+" m "+str(FibreType)+" with "+str(Wavelength)+" nm LED") 
    if(DataType==0): plt.xlabel("Integrated charge (1e-7)")
    
    plt.ylabel("Number")
    centers = (0.5*(CurrentBins[1:]+CurrentBins[:-1]))
    pars, cov = curve_fit(lambda Data, mu, sig : scipy.stats.norm.pdf(Data, loc=mu, scale=sig), centers,CurrentN, p0=Guess)
    plt.plot(centers, scipy.stats.norm.pdf(centers,*pars), 'k--',linewidth = 2, label='fit')
    Mu = pars[0]
    Mu_Uncertainty = np.sqrt(cov[0,0])
    plt.legend()
    print("For "+str(FibreLength)+" m "+str(FibreType)+" with "+str(Wavelength)+" nm LED")
    print("Fit: %0.2f +- %0.2f" %(Mu,Mu_Uncertainty))
    ## Mean value of an array
    Mean = (np.array(Data)).mean()
    print("Mean of data array = ",Mean)
    #plt.show()
    return Mu,Mu_Uncertainty,Mean

def GetSinglePhotonPeak():
    #THIS MUST BE FIXED
    # fig4=plt.figure()
    # alpha=0.5
    # colour='red'    
    # CurrentNSP,CurrentBinsSP,_=plt.hist(SinglePhotonPeakData,bins=30,density = True,range = (0,25),alpha=alpha,color=colour,label='Data')
    # plt.title("Single Photon Peak Finding") 
    # plt.xlim(0,20)
    # plt.xlabel("Peak Voltage (mV)")
    
    # from scipy.optimize import curve_fit
    # from scipy.signal import find_peaks
    # import scipy.stats
    
    # centers = (0.5*(CurrentBinsSP[1:]+CurrentBinsSP[:-1]))
    # pars, cov = curve_fit(lambda SinglePhotonPeakData, mu, sig : scipy.stats.norm.pdf(SinglePhotonPeakData, loc=mu, scale=sig), centers,CurrentNSP, p0=[2,5])
    # plt.plot(centers, scipy.stats.norm.pdf(centers,*pars), 'k--',linewidth = 2, label='fit')
    # Mu = pars[0]
    # Mu_Uncertainty = np.sqrt(cov[0,0])
    
    # print("Fit: %0.2f +- %0.2f" %(Mu,Mu_Uncertainty))
    # # Mean value of the array
    # Mean = (np.array(SinglePhotonPeakData)).mean()
    # print("Mean of data array = ",Mean)
    return 500