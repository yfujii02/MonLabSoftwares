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
ThresholdFreq = 0.32e8
IntegrationWidth=40
NumMAv=5 # Number of moving average points
PedestalRange=90

def GetFileList(DataPath,Date,FibreType,FibreLength,LEDWavelength,Voltage):
    #Example: DataPath = r'C:\Users\smdek2\Documents\SiPMTests\PlasticFibreTests\Sep17_InitialGlassFibreTests\'
    #         Date = Sep17
    #         Voltage = 42
    #         FibreLength = 10  
    #         FibreType = glass
    #         LEDWavelength = 405
    FilePaths = str(DataPath)+str(Date)+'_hv'+str(Voltage)+'_'+str(FibreLength)+'m_'+str(FibreType)+'_LED'+str(LEDWavelength)+'*.npy'
    #print("File Path = ",FilePaths)
    FileList=glob.glob(FilePaths)    
    return FileList

#### Get waveforms from the npy file
def GetWaveformData(Filename):
    FileData = np.array(np.load(Filename,allow_pickle=True))            
    WaveformData = FileData[6]
    return WaveformData

#### Compatible to the original peak search function written by SamD
def PeakSearchOrig(wf,peakSearchRange,peakThreshold,baseLevel,maxIndex,startIndex,endIndex):
    waveform = wf[peakSearchRange[0]:peakSearchRange[1]]
    if waveform.max()<peakSearchThreshold:
        ### Return if no peak found...
        return False
    maxIndices=np.argmax(waveform)
    maxIndex=maxIndices[0] ## Ignore after pulses now
    for i in range(maxIndex,0,-1):
        if(waveform[i]<=baseLevel):
            startIndex=i
            break
    for i in range(maxIndex,len(waveform),1)
        if(waveform[i]<=baseLevel):
            endIndex=i
            break
    if (startIndex<0 or endIndex<0):
        return False
    startIndex = startIndex+peakSearchRange[0]
    endIndex   = endIndex  +peakSearchRange[0]
    return True

def PeakSearchSciPy(wf,heightThr,minGap):
    find_peaks(wf,height=heightThr,distance=minGap)
    return None

def GetData(WaveformData,Output,Length):
    CurrentIntegrated=[]
    CurrentMaxPeaks=[]
    CurrentStartIndices=[]
    CurrentStopIndices=[]
    
    BadEventCounter=0
    PeakPlot=0
    if(PeakPlot==1):
        fig = plt.figure()
        plt.ion()
    for j in range(int(len(WaveformData)/2)):
       
        #FilterWaveform
        MPPCSig = WaveformData[2*j,:]
        #print("Waveform size = ",len(MPPCSig)) This is 400
        moveavefilt=np.cumsum(np.insert(MPPCSig,0,0))
        MPPCSig=(moveavefilt[NumMAv:]-moveavefilt[:-NumMAv])/float(NumMAv)

        #baseLevel=np.array(MPPCSig[NumMAv:NumMAv+PedestalRange]).mean()
        #baseStd  =np.array(MPPCSig[NumMAv:NumMAv+PedestalRange]).std()
        
        if(np.max(MPPCSig)>30): PeakThreshold=20
        else: PeakThreshold = 5
        
        if(Length==1): TimeThreshold=120
        else: TimeThreshold=180
        
        #Apply FFT for easier main peak finding
        if(FFTOn==1):
            fftV = fftpack.fft(MPPCSig)
            samplefreq=fftpack.fftfreq(Samples, dt)
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
        
        PeakFound=PeakSearchOrig(MPPCSigFilt,[TimeThreshold,len(MPPCSigFilt)-1],
                                 PeakThreshold,0,MaxIndex,StartIndex,EndIndex)
        
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
            SigRange=np.linspace(StartIndex,EndIndex,1)
            
            plt.plot(MPPCSig, color='k')
            plt.plot(SigRange,MPPCSig[StartIndex,EndIndex+1], color='r')
            plt.plot(MPPCSigFilt,color='blue')
            plt.draw()
            plt.pause(0.01)
            fig.clear()
          
        CurrentStartIndices.append(StartIndex)
        CurrentStopIndices.append(EndIndex)
        
        #Numerical Integration using Simpsons Rule 
        if(StartIndex>=EndIndex): Area = 0.0
        else: Area = simps((MPPCSig)[StartIndex:EndIndex],TimeArray[StartIndex:EndIndex])
     
        CurrentIntegrated.append(Area*1e7)
        Output.append(Area*1e7)
        CurrentMaxPeaks.append(np.max(MPPCSig[StartIndex:EndIndex]))

    return 1


def GetHistogram(Data,NBins,RL,RU,Guess,FibreLength,FibreType,Wavelength,SuppressPlots):
    if(SuppressPlots==0):fig = plt.figure()
    alpha=0.5
    colour='red'    
    CurrentN,CurrentBins,_=plt.hist(Data,bins=NBins,density = True,range = (RL,RU),alpha=alpha,color=colour,label='Data')
    plt.title("IC: "+str(FibreLength)+" m "+str(FibreType)+" with "+str(Wavelength)+" nm LED") 
    plt.xlim(RL, RU)
    plt.xlabel("Integrated charge (1e-7)")
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
