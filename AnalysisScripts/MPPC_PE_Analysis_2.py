# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:17:01 2021
Photon Efficiency Analysis for MPPCs
Last Updated: Oct 14 2021
@author: smdek2
"""

import numpy as np
import MonLabAnalysisFunctions21 as MLA
import matplotlib.pyplot as plt

from scipy import fftpack
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import simps

import glob 
import math
import sys

#Flags for various items
CheckSingleDataSet = 1

SingleDataFolderPath=r'/home/comet/work/pico/ScintTestOct12/' #path to singular dataset
SingleDataFolderPath2=r'/home/comet/work/pico/StrontiumTest_Oct12/' #path to singular dataset
#SingleDataFolderPath=r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct5\LEDTest_ChB1-A2_Oct5_3\41p8V\\' #path to singular dataset
FileString = "*" 
FileString2 = "*Sr90"

SingleVb = 42.0 #bias voltage of said dataset 1 = 41.0 V, 2 = 41.3 V, 3 = 41.5 V, 4 = 41.8 V, terrible idea should just be value...
if(SingleVb==1):
   ChanV = '41.0'
   colour='r'
elif(SingleVb==2):
   ChanV = '41.3'
   colour='b'
elif(SingleVb==3):
   ChanV = '41.5'
   colour='g'
elif(SingleVb==4):
   ChanV = '41.8'
   colour='purple'
else: 
   ChanV = str(SingleVb)
   colour = 'purple'


TempString = "Scint Cosmic Rays" #String used to label headings
TempString2 = "Scint Sr90 + Cosmic Rays"
#TempString = "Four Channel B1|A1|B2|A2" #String used to label headings

#For multiple datasets
DataCollectFlag = 0 #Set to zero if not changing anything about peak collection
subplot=0 #Set to plot all histograms on one figure

#Data Type
DataTypeFlag = 0 # 0 = Four Waveform Channel, 1 = LED Test

#Stop Flags
ReturnFlag=0 #flag for exiting program on some conditions - will set itself to 1 if for example less than two peaks in histogram
PeakStop=0 #stop after peak finding
HistStop=0 #stop after histogram fitting
PEStop = 0 #stop after photon efficiency (unneccessary)

#Troubleshooting plotting flags
TFlag_Peaks=0#set to 1 for more plotting in peak height extraction when troubleshooting
TFlag_Hist = 0 #set to 1 for more plotting in histogram/fitting when troubleshooting
TFlag_PE = 0 #set to 1 for more plotting in photon efficiency when troubleshooting

#Other plotting flags
HistPeakPlotFlag=1 #Plot scipy peak markers
HistFitPlotFlag=1 #plot fit to histogram

#Minimum pulse peak extraction settings
PeakThresholding = 0 #Set to 1 if you want to apply some threshold to minimum peak height in data
PeakThreshold = 3 #Threshold can be used in extracting pulse heights (mV)
PlotChannelB = 0 #Set to 1 if you also want to see trigger signal side by side with peaks


SigLower = 87 #lower limit on peak window (min = 0) (Window used on Oct 1 data was [250:330])
SigUpper = 102 #upper limit on peak window (max = 400)
# SigLower = 250 #lower limit on peak window (min = 0) (Window used on Oct 1 data was [250:330])
# SigUpper = 330 #upper limit on peak window (max = 400)



NumMAv_PeakFinding=20 #Moving average threshold applied to peak finding
TitleString='['+str(SigLower)+':'+str(SigUpper)+']' #for plot title if TFlag_Peaks=1
CutEdges=5 #cutting edges off pulses to account for moving average filter effects
IntegrateFlag = 0 #set to 1 if want to integrate instead of take minimum between peak window

if(IntegrateFlag==1): XlabelString = "Integral (ns mV)"
else: XlabelString = "Peak Voltage (mV)"


if(IntegrateFlag==1):
    NBins = 200
    RU = 250
    RL=-1750
else: 
    NBins = 180 #Histogram bins used (Used 100 for Oct 1)(Used 300 Oct12)
    RU  = 50 #Upper limit of histograms (peak voltage in mV) (used 45 for Oct 1 Data)
    RL = 5 #Lower limit of histograms (peak voltage in mV) (used -5 for Oct 1 Data)
    
N_peaks = 2 #Number of peaks to fit in histogram
MoveAveFlag = 0 #Flag for applying a moving average filter to the histogram
NumMAv_Hist=2
pdist = 8 #minimum index distance allowed between peaks when using the scipy peak find for initial fit guessing
threshold = 450 #minimum count height allowed for peaks when using the scipy peak find for initial fit guessing
Vb = [41.0,41.3,41.5] #bias voltages used when taking data
Vover = 2.7 #Overvoltage used on datasheet which can be subtracted from operational voltage to obtain breakdown voltage
VBreakDown = np.array([41.26,41.26,41.29,41.25,41.21])-Vover #Breakdown voltage for each channel from datasheet. Ordering is Ch [A3,A4,B1,C4,D2]
#VBreakDown = VBreakDown*0.0 #Use this if you want to view raw data trends for 1 p.e. and pedestal vs bias voltage 

def FFT(Data, iData):
    #OrigData=Data
    #Data = Data[175:]
    
    Samples = Data.size 
    dt = (iData[2]-iData[1])*10**-9
    #fftT = np.linspace(0.0, 1.0/(2.0*dt), Samples//2)
    fftT = fftpack.fftfreq(Samples,dt)[:Samples//2]
    ThresholdFreq = 0.2e8
    #ThresholdLowFreq = 0.5e8
    
    fftV = fftpack.fft(Data)   
    samplefreq=fftpack.fftfreq(Samples, dt)
    #plt.figure()
    #plt.plot(fftT,2.0/Samples*np.abs(fftV[0:Samples//2]))
    #print(len(2.0/Samples*np.abs(fftV[0:Samples//2])))
    Copy = fftV.copy()
    Copy[np.abs(samplefreq)>ThresholdFreq]=0
    #Copy[np.abs(samplefreq)<ThresholdLowFreq]=0
    
    filteredsig=fftpack.ifft(Copy)
    Signal = filteredsig.real
    #plt.figure()
    #plt.plot(iData[175:],Data)
    #plt.plot(iData[175:],Signal)
    Freq=np.array(2.0/Samples*np.abs(fftV[0:Samples//2]))
    iFreq = fftT
    return Signal#, Freq, iFreq

NoiseFlaggedEvents = []
TotalNoiseEvents = 0
TotalEvents=0
PlotWaveformFlag = 0

def RemoveNoiseEvents(Data,iData):
    global TotalNoiseEvents
    Out = []
    iOut=[]
    for i in range(len(Data)):
        Flag = 0
        for j in range(len(NoiseFlaggedEvents)):
            
            if(i==NoiseFlaggedEvents[j]):
                Flag = 1
                TotalNoiseEvents+=1
                break
            
        if(Flag==0):
            Out.append(Data[i])
            iOut.append(iData[i])
    return Out,iOut

def PlotWaveforms(Data,Peaks,iPeaks,NumberChannels,Channel):
    if(PlotWaveformFlag==0): return
    global NoiseFlaggedEvents
    plt.figure()
    plt.xlabel("Time (ns)")
    plt.ylabel("Voltage (mV)")
    plt.title("Channel "+str(Channel))
    #print(int(len(Data[:,0])/NumberChannels))
    #print(len(Peaks))
    #print(NoiseFlaggedEvents)
    for i in range(int(len(Data[:,0])/NumberChannels)):
        Flag = 0
        for j in range(len(NoiseFlaggedEvents)):
            if(i==NoiseFlaggedEvents[j]):
                Flag = 1
                break
        if(Flag==0):
            ChData = (Data[NumberChannels*i+Channel,:])
            
            #Apply moving average filter to waveform
            moveavefilt=np.cumsum(np.insert(ChData,0,0))
            ChData=(moveavefilt[NumMAv_PeakFinding:]-moveavefilt[:-NumMAv_PeakFinding])/float(NumMAv_PeakFinding)
            ChData = ChData[CutEdges:ChData.size-CutEdges]
            ChTime = np.linspace(0,len(ChData)*0.8,len(ChData))
            #plt.plot(ChTime[SigLower:SigUpper],ChData[SigLower:SigUpper],alpha=0.1)
            plt.plot(ChTime,ChData,alpha=0.1)
            plt.scatter(ChTime[iPeaks[i]],-Peaks[i],marker='x')
    plt.show()
    return
        
        
def ExtractPeaks(Channel, NumberChannels, Data, PeakType, DigProcessing):
    #Channel: A=0, B=1,C=2,D=3
    #NumberChannels: Total number of channels in this dataset
    #Data: File waveform data
    #PeakType: Peak = 0, Integrated = 1
    #DigProcessing: Fourier transform processing 0/n or 1/y
    global NoiseFlaggedEvents
    Output=[]
    iOutput=[]
    iOutVal=0
    #Frequencies = np.zeros(185)
    #plt.figure()
    for j in range(int(len(Data[:,0])/NumberChannels)):
        ChData = (Data[NumberChannels*j+Channel,:])
        #Apply moving average filter to waveform
        moveavefilt=np.cumsum(np.insert(ChData,0,0))
        ChData=(moveavefilt[NumMAv_PeakFinding:]-moveavefilt[:-NumMAv_PeakFinding])/float(NumMAv_PeakFinding)
        ChData = ChData[CutEdges:ChData.size-CutEdges]
        
        ChTime = np.linspace(0,len(ChData)*0.8,len(ChData))
        
        if(DigProcessing==1):
            ChData = FFT(ChData,ChTime)
            #Frequencies += Freq
            
        #Output is either integrated charge or peak voltage value
        if(PeakType==0):
            OutVal = np.min(ChData)
        elif(PeakType==1):
            OutVal = simps(ChData,ChTime)
        iOutVal = np.argmin(ChData)
        
        #print(iOutVal)
        #print(SigLower)
        #print(SigUpper)
        #Tag Events for Deletion Afterwards
        if(PeakThresholding==1 and OutVal>=-PeakThreshold): NoiseFlaggedEvents.append(j)
        
        elif(np.max(ChData)>10): NoiseFlaggedEvents.append(j)
        
        elif(np.min(ChData[:65])<-5): NoiseFlaggedEvents.append(j)
        
        elif(iOutVal<SigLower or iOutVal>SigUpper): NoiseFlaggedEvents.append(j)
        
        #elif(iOutVal>=SigLower and iOutVal<=SigUpper and OutVal>-9): NoiseFlaggedEvents.append(j)
        #print("#####")
                 
        #Need to append all and then just note down which events are noisy to remove from all channels in one go
        Output.append(-OutVal)
        iOutput.append(iOutVal)
                
    #plt.figure()
    #plt.plot(iFreq,Frequencies)
    return np.array(Output),np.array(iOutput)

def PlotLEDTestHist(FPath,FString):
    global NoiseFlaggedEvents
    global TotalNoiseEvents
    global TotalEvents
    FilePaths = str(FPath)+str(FString)+'*.npy'
    FileList=glob.glob(FilePaths)
    print(FileList)
    FFT = 0
    FileOutput=[]
    FileBOutput=[]
    NCh = 2
    if(DataTypeFlag==0): 
        NCh = 4
        FileCOutput=[]
        FileDOutput=[]
        if(TFlag_Peaks==1):
            fig,([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2)
            ax1.set_title('Ch A')
            ax3.set_xlabel('Time (ns)')
            ax1.set_ylabel('Voltage (mV)')
            ax2.set_title('Ch B')
            ax4.set_xlabel('Time (ns)')
            ax3.set_ylabel('Voltage (mV)')
    TriggerRates=[]    
    for i in range(len(FileList)):
        NoiseFlaggedEvents=[]
        WaveFormData, HeaderInfo = MLA.GetWaveformData(FileList[i])
        print("Current File: ",FileList[i])
        print(HeaderInfo)
        TriggerRates.append((HeaderInfo[0][4]/(HeaderInfo[0][3]-HeaderInfo[0][2])))
        print("Trigger rate = ",(HeaderInfo[0][4]/(HeaderInfo[0][3]-HeaderInfo[0][2])))
        
        ChAPeaks, iChAPeaks = ExtractPeaks(0,NCh,WaveFormData,0,FFT)
        
        
        if(DataTypeFlag==0):
            ChBPeaks, iChBPeaks = ExtractPeaks(1,NCh,WaveFormData,0,FFT)
            ChCPeaks, iChCPeaks = ExtractPeaks(2,NCh,WaveFormData,0,FFT)
            ChDPeaks, iChDPeaks = ExtractPeaks(3,NCh,WaveFormData,0,FFT)
            
            #Optional plotting
            PlotWaveforms(WaveFormData,ChAPeaks,iChAPeaks,NCh,0)
            PlotWaveforms(WaveFormData,ChBPeaks,iChBPeaks,NCh,1)
            PlotWaveforms(WaveFormData,ChCPeaks,iChCPeaks,NCh,2)
            PlotWaveforms(WaveFormData,ChDPeaks,iChDPeaks,NCh,3)
            
            TotalEvents+=len(WaveFormData)/NCh
            
            ChAPeaks,iChAPeaks = RemoveNoiseEvents(ChAPeaks,iChAPeaks)
            ChBPeaks,iChBPeaks = RemoveNoiseEvents(ChBPeaks,iChBPeaks)    
            ChCPeaks,iChCPeaks = RemoveNoiseEvents(ChCPeaks,iChCPeaks)    
            ChDPeaks,iChDPeaks = RemoveNoiseEvents(ChDPeaks,iChDPeaks)   
            
            if(TFlag_Peaks==1):
                ax1.scatter(0.8*(iChAPeaks+SigLower), -1*ChAPeaks,marker='x')
                ax2.scatter(0.8*(iChBPeaks+SigLower), -1*ChBPeaks,marker='x')
                ax3.scatter(0.8*(iChCPeaks+SigLower), -1*ChCPeaks,marker='x')
                ax4.scatter(0.8*(iChDPeaks+SigLower), -1*ChDPeaks,marker='x')
            
            FileOutput.append(np.array(ChAPeaks)) #Append pulse peaks
            FileBOutput.append(np.array(ChBPeaks)) #Append pulse peaks
            FileCOutput.append(np.array(ChCPeaks)) #Append pulse peaks
            FileDOutput.append(np.array(ChDPeaks)) #Append pulse peaks
            
        elif(DataTypeFlag==1 and PlotChannelB==1):
            ChBPeaks, iChBPeaks = ExtractPeaks(1,NCh,WaveFormData,0,0)
            FileOutput.append(np.array(ChAPeaks)) #Append pulse peaks
            FileBOutput.append(np.array(ChBPeaks)) #Append pulse peaks
    
    Output=np.concatenate(np.array(FileOutput),axis=0)
    if(DataTypeFlag==0):
        OutputB=np.concatenate(np.array(FileBOutput),axis=0)
        OutputC=np.concatenate(np.array(FileCOutput),axis=0)
        OutputD=np.concatenate(np.array(FileDOutput),axis=0)
        
        return Output, OutputB, OutputC, OutputD, np.array(TriggerRates)
    else: return Output, np.array(TriggerRates)   


def HistogramFit(x,*params):
    y = np.zeros_like(x)
    for i in range(0,len(params),3):
        mean = params[i]
        amplitude = params[i+1]
        sigma = params[i+2]
        y = y + amplitude * np.exp(-((x-mean)/sigma)**2)
    return y

def PlottingFit(axis,channeldata,channel_Vb,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak,SP_Peak,uPed,uSP):
    global ReturnFlag
    #channel_Vb = 1, 2 or 3
    if(channel_Vb==1):
       ChanV = '41.0'
       colour='r'
    elif(channel_Vb==2):
       ChanV = '41.3'
       colour='b'
    elif(channel_Vb==3):
       ChanV = '41.5'
       colour='g'
    elif(channel_Vb==4):
       ChanV = '41.8'
       colour='purple'
    else: 
       ChanV = str(SingleVb)
       colour = 'purple'
    #Ch Name
    if(axis==0):ChName = TempString
    elif(axis==1):ChName = "Ch B1"
    elif(axis==2):ChName= "Ch C4"
    elif(axis==3):ChName= "Ch D2"
    elif(axis==4):ChName= "Ch A4"
    elif(axis==5):ChName= "Ch A3"
    elif(axis==6):ChName= "Ch B1A2"
    
    
    
    if(axis==3):NumMAv=3
    else: NumMAv=NumMAv_Hist
    
    PeakMarkerSize=1 
    
    if(subplot==1):
        if(axis==1):
            CurrentN,CurrentBins,_=ax1.hist(channeldata,range=[RL,RU],bins=NBins,color=colour,alpha=0.5,label='Vb = '+ ChanV + ' V')
            
            if(MoveAveFlag==1):
                moveavefilt=np.cumsum(np.insert(CurrentN,0,0))
                CurrentN=(moveavefilt[NumMAv:]-moveavefilt[:-NumMAv])/float(NumMAv)
                CurrentN = CurrentN[2:CurrentN.size-2]
                CurrentBins=CurrentBins[:CurrentN.size]
                #plt.bar(CurrentBins,CurrentN)
                    
            peaks, _ = find_peaks(CurrentN, distance=pdist, height=threshold)
            ax1.scatter(CurrentBins[peaks], CurrentN[peaks],s=PeakMarkerSize,color='k',marker='x')
            
        elif(axis==2):
            CurrentN,CurrentBins,_=ax2.hist(channeldata,range=[RL,RU],bins=NBins,color=colour,alpha=0.5,label='Vb = '+ ChanV + ' V')
            
            if(MoveAveFlag==1):
                moveavefilt=np.cumsum(np.insert(CurrentN,0,0))
                CurrentN=(moveavefilt[NumMAv:]-moveavefilt[:-NumMAv])/float(NumMAv)
                CurrentN = CurrentN[2:CurrentN.size-2]
                CurrentBins=CurrentBins[:CurrentN.size]
                #plt.bar(CurrentBins,CurrentN)
            
            peaks, _ = find_peaks(CurrentN, distance=pdist,height=threshold)
            ax2.scatter(CurrentBins[peaks], CurrentN[peaks],s=PeakMarkerSize,color='k',marker='x')
    
        elif(axis==3):
            CurrentN,CurrentBins,_=ax3.hist(channeldata,range=[RL,RU],bins=NBins,color=colour,alpha=0.5,label='Vb = '+ ChanV + ' V')
            
            if(MoveAveFlag==1):
                moveavefilt=np.cumsum(np.insert(CurrentN,0,0))
                CurrentN=(moveavefilt[NumMAv:]-moveavefilt[:-NumMAv])/float(NumMAv)
                CurrentN = CurrentN[2:CurrentN.size-2]
                CurrentBins=CurrentBins[:CurrentN.size]
                #plt.bar(CurrentBins,CurrentN)
            
            peaks, _ = find_peaks(CurrentN, distance=pdist,height=threshold)
            ax3.scatter(CurrentBins[peaks], CurrentN[peaks],s=PeakMarkerSize,color='k',marker='x')
       
    
        elif(axis==4):
            CurrentN,CurrentBins,_=ax4.hist(channeldata,range=[RL,RU],bins=NBins,color=colour,alpha=0.5,label='Vb = '+ ChanV + ' V')
            
            if(MoveAveFlag==1):
                moveavefilt=np.cumsum(np.insert(CurrentN,0,0))
                CurrentN=(moveavefilt[NumMAv:]-moveavefilt[:-NumMAv])/float(NumMAv)
                CurrentN = CurrentN[2:CurrentN.size-2]
                CurrentBins=CurrentBins[:CurrentN.size]
                #plt.bar(CurrentBins,CurrentN)
            
            peaks, _ = find_peaks(CurrentN, distance=pdist,height=threshold)
            ax4.scatter(CurrentBins[peaks], CurrentN[peaks],s=PeakMarkerSize,color='k',marker='x')
    
        elif(axis==5):
            CurrentN,CurrentBins,_=ax5.hist(channeldata,range=[RL,RU],bins=NBins,color=colour,alpha=0.5,label='Vb = '+ ChanV + ' V')
            
            if(MoveAveFlag==1):
                moveavefilt=np.cumsum(np.insert(CurrentN,0,0))
                CurrentN=(moveavefilt[NumMAv:]-moveavefilt[:-NumMAv])/float(NumMAv)
                CurrentN = CurrentN[2:CurrentN.size-2]
                CurrentBins=CurrentBins[:CurrentN.size]
                #plt.bar(CurrentBins,CurrentN)
                
            peaks, _ = find_peaks(CurrentN, distance=pdist,height=threshold)
            ax5.scatter(CurrentBins[peaks], CurrentN[peaks],s=PeakMarkerSize,color='k',marker='x')
    
        elif(axis==6):
           CurrentN,CurrentBins,_=ax2.hist(channeldata,range=[RL,RU],bins=NBins,color=colour,alpha=0.5,label='Vb = '+ ChanV + ' V')
            
           if(MoveAveFlag==1):
                moveavefilt=np.cumsum(np.insert(CurrentN,0,0))
                CurrentN=(moveavefilt[NumMAv:]-moveavefilt[:-NumMAv])/float(NumMAv)
                CurrentN = CurrentN[2:CurrentN.size-2]
                CurrentBins=CurrentBins[:CurrentN.size]
                #plt.bar(CurrentBins,CurrentN)
            
           peaks, _ = find_peaks(CurrentN, distance=pdist,height=threshold)
           ax6.scatter(CurrentBins[peaks], CurrentN[peaks],s=PeakMarkerSize,color='k',marker='x')
    else:
        plt.figure()
        CurrentN,CurrentBins,_=plt.hist(channeldata,range=[RL,RU],bins=NBins,color=colour,alpha=0.5,label='Vb = '+ ChanV + ' V')
        plt.title(ChName+" Vb = "+ChanV)
        plt.xlabel(XlabelString)
        plt.ylabel("Count")
        
       
   
        if(MoveAveFlag==1):
                moveavefilt=np.cumsum(np.insert(CurrentN,0,0))
                CurrentN=(moveavefilt[NumMAv:]-moveavefilt[:-NumMAv])/float(NumMAv)
                CurrentN = CurrentN[2:CurrentN.size-2]
                CurrentBins=CurrentBins[:CurrentN.size]
                
    
        peaks, _ = find_peaks(CurrentN, distance=pdist,height=threshold)
        if(HistPeakPlotFlag==1):
            if(MoveAveFlag==1):
                plt.figure()
                plt.bar(CurrentBins,CurrentN) 
                plt.scatter(CurrentBins[peaks], CurrentN[peaks],s=20,color='k',marker='x')
                plt.title(ChName+" Vb = "+ChanV+" Peaks")
                plt.xlabel(XlabelString)
                plt.ylabel("Count")
            else: plt.scatter(CurrentBins[peaks], CurrentN[peaks],s=20,color='k',marker='x')

    #scipy Peak guesses
    PeakX = CurrentBins[peaks]
   
    PeakY = CurrentN[peaks]
    
    PeakI = peaks
    
    if(PeakX.size<2):
        print('Less than 2 peaks - NO GOOD!')
        ReturnFlag = 1
        a=[]
        return a, a, a, a, np.array(CurrentBins), np.array(CurrentN), a, a  
    

    
    #Guess of fitting window  
    PEWidth = PeakX[1]-PeakX[0]

    PEWidthI = PeakI[1]-PeakI[0]
    
    FitCutGuessI = int(PeakI[0]+round((N_peaks+0.5)*PEWidthI))
    
    WindowAdjustment = 2
    
    XWindow = CurrentBins[:FitCutGuessI+WindowAdjustment] 
    YWindow = CurrentN[:FitCutGuessI+WindowAdjustment]
    

    
    #Fitting Guess in form params = [mean,amplitude, sigma] for each gaussian
    guess_pedestal = PeakX[0] 
   
    guess_params = [guess_pedestal,PeakY[0],PEWidth/2]
    for i in range(N_peaks):
        #Initial peak height guess is that it approximately halves for each
        guess_params += [guess_pedestal+(i+1)*PEWidth,PeakY[1]/(2**i),PEWidth/2]
    
    popt, pcov = curve_fit(HistogramFit,XWindow,YWindow,p0=guess_params)
    
    #Second Fit using first fit parameters
    XWindow = CurrentBins[:np.searchsorted(CurrentBins,(popt[N_peaks*3]+popt[N_peaks*3+2]))] 
    YWindow = CurrentN[:np.searchsorted(CurrentBins,(popt[N_peaks*3]+popt[N_peaks*3+2]))]
    guess_pedestal = popt[0] 
   
    guess_params = [guess_pedestal,popt[1],popt[2]]
    for i in range(N_peaks):
        #Initial peak height guess is that it approximately halves for each
        guess_params += [popt[(i+1)*3],popt[(i+1)*3+1],popt[(i+1)*3+2]]
    
        
    popt, pcov = curve_fit(HistogramFit,XWindow,YWindow,p0=guess_params)
    print("#########")
    #print(popt)
    #print(pcov)
    p_sigma = np.sqrt(np.diag(pcov))
    fit = HistogramFit(XWindow,*popt)
    lw = 1
    if(HistFitPlotFlag==1):
        if(subplot==1):
            if(axis==1): ax1.plot(XWindow,fit,'k--',linewidth=lw)    
            elif(axis==2): ax2.plot(XWindow,fit,'k--',linewidth=lw)  
            elif(axis==3): ax3.plot(XWindow,fit,'k--',linewidth=lw)  
            elif(axis==4): ax4.plot(XWindow,fit,'k--',linewidth=lw)  
            elif(axis==5): ax5.plot(XWindow,fit,'k--',linewidth=lw)  
            elif(axis==6): ax6.plot(XWindow,fit,'k--',linewidth=lw)  
        else:
            plt.plot(XWindow,fit,'k--',linewidth=lw)
    
    #Print Pedestal and 1 p.e. values
    print(ChName+' at Vb = '+ChanV)
    if(IntegrateFlag==1):
          print("Pedestal: Charge = %.2f +- %.2f" %(popt[0],p_sigma[0]))
          print("1 p.e. : Charge = %.2f +- %.2f" %(popt[3],p_sigma[3]))
          if(N_peaks>1): print("2 p.e. : Charge = %.2f +- %.2f" %(popt[6],p_sigma[6]))
    else:
        print("Pedestal: V = %.2f +- %.2f" %(popt[0],p_sigma[0]))
        print("1 p.e. : V = %.2f +- %.2f" %(popt[3],p_sigma[3]))
        if(N_peaks>1): print("2 p.e. : V = %.2f +- %.2f" %(popt[6],p_sigma[6]))
        
    SP_Peak.append(popt[3])
    Ped_Peak.append(popt[0])
    uSP.append(p_sigma[3])
    uPed.append(p_sigma[0])
    
    if(TFlag_Hist==1):
        plt.figure()
        plt.plot(XWindow,YWindow)
        plt.show()
        
    FitOutput = popt
    uFitOutput = p_sigma
    return Ped_Peak, SP_Peak, uPed, uSP, CurrentBins, CurrentN, FitOutput, uFitOutput    

def Poisson(n, mu):
    return ((mu**n)*np.exp(-mu))/(math.factorial(n))
    
def PhotonEfficiencyCalc(ch_bins,ch_counts,PEWidths,Pedestals,channel_Vb,axis,FitO,uFitO):
    #channel_Vb = 1, 2 or 3
    if(channel_Vb==1):
       ChanV = '41.0'
       colour='r'
    elif(channel_Vb==2):
       ChanV = '41.3'
       colour='b'
    elif(channel_Vb==3):
       ChanV = '41.5'
       colour='g'
    elif(channel_Vb==4):
       ChanV = '41.8'
       colour='purple'
    else: 
       ChanV = str(SingleVb)
       colour = 'purple'
    
    #Ch Name
    if(axis==0):ChName = TempString
    elif(axis==1):ChName = "Ch B1"
    elif(axis==2):ChName= "Ch C4"
    elif(axis==3):ChName= "Ch D2"
    elif(axis==4):ChName= "Ch A4"
    elif(axis==5):ChName= "Ch A3"
    elif(axis==6):ChName = "Ch B1-A2"
    
    #Input data of SPmPed of a channel
    PEWidth = PEWidths[channel_Vb-1]
    Pedestal = Pedestals[channel_Vb-1]
    
    XCut = ch_bins[:np.searchsorted(ch_bins,(Pedestal+0.5*PEWidth))]
    YCut = ch_counts[:np.searchsorted(ch_bins,(Pedestal+0.5*PEWidth))]
    
    #Int = np.sum(YCut*XCut)
    #print('Integral of pedestal = ',Int)
    #Nped = np.sum(YCut)
    #print('Nped = ',Nped)
    #N_ped should integrate gaussian fit to pedestal up to 3 sigma
    #XCutVal = FitO[0]+3*FitO[2] #mu+3 sigma
    #Nped = FitO[1]*(1+math.erf(XCutVal-FitO[0]/(FitO[2])))
    
    #Using int(a exp(-(x-b)^2/2c^2)) dx between +- inf = sqrt(2)*a*|c|*sqrt(pi)
    #2c^2 = FitO[2] -> sqrt(FitO[2]) = sqrt(2)|c|
    Nped = np.sqrt(FitO[0])*FitO[1]*np.sqrt(math.pi)
    
    
    Ntot = np.sum(ch_counts)
    print('Ntot = ',Ntot)
    Mean = -np.log(Nped/Ntot)
    print("Poisson mean = ",Mean)    
    
    Nph=[]
    N= 10
    for i in range(N):
        Pois=0
        for j in range(i):
            Pois += Poisson(j,Mean)    
        Nph.append(Ntot*Pois)
    
    if(TFlag_PE==1):
        plt.figure()
        plt.plot(XCut,YCut)
        plt.show()
    
    print("####")
    print(ChName+' '+ChanV)
    print(Nph)
    Mean = round(Mean,2)
    plt.scatter(np.array(list(range(1,N+1))),Nph/Ntot,color=colour,label=ChanV+', mu = '+str(Mean)+' ph')
    
    return Nph

if(CheckSingleDataSet==1):
      
    SP_Peak=[]
    Ped_Peak=[]
    uSP=[]
    uPed=[]
    
    if(DataTypeFlag==1):
        ChSingle = PlotLEDTestHist(SingleDataFolderPath,FileString)
    else:
        ChSingle,ChSingleB,ChSingleC,ChSingleD,TriggerRates1 = PlotLEDTestHist(SingleDataFolderPath,FileString)
        a = TotalNoiseEvents/TotalEvents
        TotalNoiseEvents=0
        TotalEvents=0
        ChSingle2,ChSingleB2,ChSingleC2,ChSingleD2,TriggerRates2 = PlotLEDTestHist(SingleDataFolderPath2,FileString2)
        print("Discarded fraction events (cosmic) = ",a)
        print("Discarded fraction events (cosmic+Sr90) = ",TotalNoiseEvents/TotalEvents)
    if(PeakStop==1): sys.exit()
   
    Ped_Peak, SP_Peak, uPed, uSP, sCurrentBins, sCurrentN, sFit, suFit = PlottingFit(0,ChSingle+ChSingleB+ChSingleC+ChSingleD,SingleVb,4*RU,RL,NBins,pdist,threshold,0,Ped_Peak,SP_Peak, uPed, uSP)
    
    Ped_Peak2, SP_Peak2, uPed2, uSP2, sCurrentBins2, sCurrentN2, sFit2, suFit2 = PlottingFit(0,ChSingle2+ChSingleB2+ChSingleC2+ChSingleD2,SingleVb,4*RU,RL,NBins,pdist,threshold,0,Ped_Peak,SP_Peak, uPed, uSP)
   
    sCurrentN = np.mean(TriggerRates1)*sCurrentN/np.sum(sCurrentN)
    print("Mean cosmic trigger rate = ", np.mean(TriggerRates1))
    sCurrentN2 = np.mean(TriggerRates2)*sCurrentN2/np.sum(sCurrentN2)
    print("Mean cosmic + Sr90 trigger rate = ", np.mean(TriggerRates2))
    StrontiumSpectrum = sCurrentN2-sCurrentN    
   
    #Plot histogram using bar plot
    plt.figure()
    plt.bar(sCurrentBins[:-1],StrontiumSpectrum, color='blue') 
    plt.title("Strontium Spectrum (Rough)")
    plt.xlabel(XlabelString)
    plt.ylabel("Count")    
 
    #if(ReturnFlag==1 or HistStop==1): sys.exit()
    NBins = 50
    if(DataTypeFlag==0):
        Ped_Peak, SP_Peak, uPed, uSP, sCurrentBins, sCurrentN, sFit, suFit = PlottingFit(0,ChSingle,SingleVb,RU,RL,NBins,pdist,threshold,0,Ped_Peak,SP_Peak, uPed, uSP)
        Ped_Peak, SP_Peak, uPed, uSP, sCurrentBins, sCurrentN, sFit, suFit = PlottingFit(0,ChSingleB,SingleVb,RU,RL,NBins,pdist,threshold,0,Ped_Peak,SP_Peak, uPed, uSP)
        Ped_Peak, SP_Peak, uPed, uSP, sCurrentBins, sCurrentN, sFit, suFit = PlottingFit(0,ChSingleC,SingleVb,RU,RL,NBins,pdist,threshold,0,Ped_Peak,SP_Peak, uPed, uSP)
        Ped_Peak, SP_Peak, uPed, uSP, sCurrentBins, sCurrentN, sFit, suFit = PlottingFit(0,ChSingleD,SingleVb,RU,RL,NBins,pdist,threshold,0,Ped_Peak,SP_Peak, uPed, uSP)
        
        Ped_Peak, SP_Peak, uPed, uSP, sCurrentBins, sCurrentN, sFit, suFit = PlottingFit(0,ChSingle2,SingleVb,RU,RL,NBins,pdist,threshold,0,Ped_Peak,SP_Peak, uPed, uSP)
        Ped_Peak, SP_Peak, uPed, uSP, sCurrentBins, sCurrentN, sFit, suFit = PlottingFit(0,ChSingleB2,SingleVb,RU,RL,NBins,pdist,threshold,0,Ped_Peak,SP_Peak, uPed, uSP)
        Ped_Peak, SP_Peak, uPed, uSP, sCurrentBins, sCurrentN, sFit, suFit = PlottingFit(0,ChSingleC2,SingleVb,RU,RL,NBins,pdist,threshold,0,Ped_Peak,SP_Peak, uPed, uSP)
        Ped_Peak, SP_Peak, uPed, uSP, sCurrentBins, sCurrentN, sFit, suFit = PlottingFit(0,ChSingleD2,SingleVb,RU,RL,NBins,pdist,threshold,0,Ped_Peak,SP_Peak, uPed, uSP)
    
    plt.show()    
    if(ReturnFlag==1 or HistStop==1): sys.exit()

    
    #Convert each data list to numpy arrays
    SP_Peak=np.array(SP_Peak)
    Ped_Peak=np.array(Ped_Peak)
    uPed=np.array(uPed)
    uSP=np.array(uSP)
    
    print("1 p.e. - Pedestal %.2f +- %.2f "%((SP_Peak-Ped_Peak),np.sqrt(uPed**2+uSP**2)))
    
    plt.figure()
    NphSingle = PhotonEfficiencyCalc(sCurrentBins,sCurrentN,(SP_Peak-Ped_Peak),Ped_Peak,1,0,sFit,suFit) 
    
    plt.title(TempString)
    plt.xlabel('N')
    plt.ylabel('$P(n_{ph}<=N)$')
        











else: 
    #Define lists for each channel's 1 p.e. and pedestal value as well as uncertainties
    SP_Peak_ChB1=[]
    Ped_Peak_ChB1=[]
    uSPB1=[]
    uPedB1=[]
    
    SP_Peak_ChD2=[]
    Ped_Peak_ChD2=[]
    uSPD2=[]
    uPedD2=[]
    
    SP_Peak_ChC4=[]
    Ped_Peak_ChC4=[]
    uSPC4=[]
    uPedC4=[]
    
    SP_Peak_ChA3=[]
    Ped_Peak_ChA3=[]
    uSPA3=[]
    uPedA3=[]
    
    SP_Peak_ChA4=[]
    Ped_Peak_ChA4=[]
    uSPA4 = []
    uPedA4=[]
    
    SP_Peak_ChB1A2=[]
    Ped_Peak_ChB1A2=[]
    uSPB1A2 = []
    uPedB1A2=[]
    
    
    
    if(DataCollectFlag==1):
        DataArray = []
        
        ChB1_A = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChB1_Oct1\41p0V\\')
        DataArray.append(ChB1_A)
        ChB1_B = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChB1_Oct1\41p3V\\')
        DataArray.append(ChB1_B)
        ChB1_C = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChB1_Oct1\41p5V\\')
        DataArray.append(ChB1_C)
        
        ChC4_A = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChC4_Oct1\41p0V\\')
        DataArray.append(ChC4_A)
        ChC4_B = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChC4_Oct1\41p3V\\')
        DataArray.append(ChC4_B)
        ChC4_C = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChC4_Oct1\41p5V\\')
        DataArray.append(ChC4_C)
        
        ChD2_A = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChD2_Oct1\41p0V\\')
        DataArray.append(ChD2_A)
        ChD2_B = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChD2_Oct1\41p3V\\')
        DataArray.append(ChD2_B)
        ChD2_C = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChD2_Oct1\41p5V\\')
        DataArray.append(ChD2_C)
        
        ChA4_A = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChA4_Oct1\41p0V\\')
        DataArray.append(ChA4_A)
        ChA4_B = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChA4_Oct1\41p3V\\')
        DataArray.append(ChA4_B)
        ChA4_C = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChA4_Oct1\41p5V\\')
        DataArray.append(ChA4_C)
        
        ChA3_A = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChA3_Oct1\41p0V\\')
        DataArray.append(ChA3_A)
        ChA3_B = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChA3_Oct1\41p3V\\')
        DataArray.append(ChA3_B)
        ChA3_C = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct1\LEDTest_ChA3_Oct1\41p5V\\')
        DataArray.append(ChA3_C)
        
        ChB1A2_A = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct5\LEDTest_ChB1-A2_Oct5\41p0V\\')
        DataArray.append(ChB1A2_A)
        ChB1A2_B = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct5\LEDTest_ChB1-A2_Oct5\41p3V\\')
        DataArray.append(ChB1A2_B)
        ChB1A2_C = PlotLEDTestHist(r'C:\Users\smdek2\MPPCTests2021\LED_Ch_Data_Oct5\LEDTest_ChB1-A2_Oct5\41p5V\\')
        DataArray.append(ChB1A2_C)
        
        FilePath = r'C:\Users\smdek2\MPPCTests2021\\'
        DataArray=np.transpose(DataArray)
        #Vbias => A = 41.0 V, B = 41.3 V, C = 41.5 V
        np.savetxt(FilePath+'Oct1LEDPeakData.txt', DataArray, fmt='%.18e', delimiter=' ', newline='\n', header='ChB1 A B C, ChD2 A B C, Ch C4 A B C, Ch A4 A B C, Ch A3 A B C', footer='', comments='# ')
    
    if(DataCollectFlag==0):
        FilePath = r'C:\Users\smdek2\MPPCTests2021\\'
        Dataset = np.loadtxt(FilePath+'Oct1LEDPeakData.txt')
        ChB1_A = Dataset[:,0]
        ChB1_B = Dataset[:,1]
        ChB1_C = Dataset[:,2]
        
        ChD2_A = Dataset[:,3]
        ChD2_B = Dataset[:,4]
        ChD2_C = Dataset[:,5]
        
        ChC4_A = Dataset[:,6]
        ChC4_B = Dataset[:,7]
        ChC4_C = Dataset[:,8]
        
        ChA4_A = Dataset[:,9]
        ChA4_B = Dataset[:,10]
        ChA4_C = Dataset[:,11]
        
        ChA3_A = Dataset[:,12]
        ChA3_B = Dataset[:,13]
        ChA3_C = Dataset[:,14]
        
    
    
    
    if(subplot==1):
        fig,([ax1,ax2],[ax3,ax4],[ax5,ax6]) = plt.subplots(3,2)
        #fig,ax1=plt.subplots(1,1)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
    
    
    Ped_Peak_ChB1, SP_Peak_ChB1, uPedB1, uSPB1, CurrentBinsB1A, CurrentNB1A, FitB1A, uFitB1A = PlottingFit(1,ChB1_A,1,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChB1,SP_Peak_ChB1, uPedB1, uSPB1)
    Ped_Peak_ChB1, SP_Peak_ChB1, uPedB1, uSPB1, CurrentBinsB1B, CurrentNB1B, FitB1B, suFitB1B = PlottingFit(1,ChB1_B,2,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChB1,SP_Peak_ChB1, uPedB1, uSPB1)
    Ped_Peak_ChB1, SP_Peak_ChB1, uPedB1, uSPB1, CurrentBinsB1C, CurrentNB1C, FitB1C, suFitB1C = PlottingFit(1,ChB1_C,3,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChB1,SP_Peak_ChB1, uPedB1, uSPB1)
    
    Ped_Peak_ChC4, SP_Peak_ChC4, uPedC4, uSPC4, CurrentBinsC4A, CurrentNC4A, FitC4A, uFitC4A = PlottingFit(2,ChC4_A,1,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChC4,SP_Peak_ChC4, uPedC4, uSPC4)
    Ped_Peak_ChC4, SP_Peak_ChC4, uPedC4, uSPC4, CurrentBinsC4B, CurrentNC4B, FitC4B, FitC4B = PlottingFit(2,ChC4_B,2,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChC4,SP_Peak_ChC4, uPedC4, uSPC4)
    Ped_Peak_ChC4, SP_Peak_ChC4, uPedC4, uSPC4, CurrentBinsC4C, CurrentNC4C, FitC4C, uFitC4C = PlottingFit(2,ChC4_C,3,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChC4,SP_Peak_ChC4, uPedC4, uSPC4)
    
    Ped_Peak_ChD2, SP_Peak_ChD2, uPedD2, uSPD2, CurrentBinsD2A, CurrentND2A, FitD2A, uFitD2A = PlottingFit(3,ChD2_A,1,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChD2,SP_Peak_ChD2, uPedD2, uSPD2)
    Ped_Peak_ChD2, SP_Peak_ChD2, uPedD2, uSPD2, CurrentBinsD2B, CurrentND2B, FitD2B, uFitD2B = PlottingFit(3,ChD2_B,2,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChD2,SP_Peak_ChD2, uPedD2, uSPD2)
    Ped_Peak_ChD2, SP_Peak_ChD2, uPedD2, uSPD2, CurrentBinsD2C, CurrentND2C, FitD2C, uFitD2C = PlottingFit(3,ChD2_C,3,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChD2,SP_Peak_ChD2, uPedD2, uSPD2)
    
    Ped_Peak_ChA4, SP_Peak_ChA4, uPedA4, uSPA4, CurrentBinsA4A, CurrentNA4A, FitA4A, uFitA4A = PlottingFit(4,ChA4_A,1,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChA4,SP_Peak_ChA4, uPedA4, uSPA4)
    Ped_Peak_ChA4, SP_Peak_ChA4, uPedA4, uSPA4, CurrentBinsA4B, CurrentNA4B, FitA4B, uFitA4B = PlottingFit(4,ChA4_B,2,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChA4,SP_Peak_ChA4, uPedA4, uSPA4)
    Ped_Peak_ChA4, SP_Peak_ChA4, uPedA4, uSPA4, CurrentBinsA4C, CurrentNA4C, FitA4C, uFitA4C = PlottingFit(4,ChA4_C,3,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChA4,SP_Peak_ChA4, uPedA4, uSPA4)
    
    Ped_Peak_ChA3, SP_Peak_ChA3, uPedA3, uSPA3, CurrentBinsA3A, CurrentNA3A, FitA3A, uFitA3A = PlottingFit(5,ChA3_A,1,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChA3,SP_Peak_ChA3, uPedA3, uSPA3)
    Ped_Peak_ChA3, SP_Peak_ChA3, uPedA3, uSPA3, CurrentBinsA3B, CurrentNA3B, FitA3B, uFitA3B = PlottingFit(5,ChA3_B,2,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChA3,SP_Peak_ChA3, uPedA3, uSPA3)
    Ped_Peak_ChA3, SP_Peak_ChA3, uPedA3, uSPA3, CurrentBinsA3C, CurrentNA3C, FitA3C, uFitA3C = PlottingFit(5,ChA3_C,3,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChA3,SP_Peak_ChA3, uPedA3, uSPA3)    
    
    Ped_Peak_ChB1A2, SP_Peak_ChB1A2, uPedB1A2, uSPB1A2, CurrentBinsB1A2A, CurrentNB1A2A, FitB1A2A, uFitB1A2A = PlottingFit(6,ChB1A2_A,1,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChB1A2,SP_Peak_ChB1A2, uPedB1A2, uSPB1A2)
    Ped_Peak_ChB1A2, SP_Peak_ChB1A2, uPedB1A2, uSPB1A2, CurrentBinsB1A2B, CurrentNB1A2B, FitB1A2B, uFitB1A2B = PlottingFit(6,ChB1A2_B,2,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChB1A2,SP_Peak_ChB1A2, uPedB1A2, uSPB1A2)
    Ped_Peak_ChB1A2, SP_Peak_ChB1A2, uPedB1A2, uSPB1A2, CurrentBinsB1A2C, CurrentNB1A2C, FitB1A2C, uFitB1A2C = PlottingFit(6,ChB1A2_C,3,RU,RL,NBins,pdist,threshold,subplot,Ped_Peak_ChB1A2,SP_Peak_ChB1A2, uPedB1A2, uSPB1A2)    
    
    
    if(subplot==1):
    
        ax1.set_title("Ch B1")
        ax2.set_title("Ch C4")
        ax3.set_title("Ch D2")
        ax4.set_title("Ch A4")
        ax5.set_title("Ch A3")
        ax6.set_title("Ch B1-A2")
        
        
        fig.suptitle("LED Tests Oct 1")
        ax5.set_xlabel(XlabelString)
        ax4.set_xlabel(XlabelString)
        
        ax1.set_ylabel('Count')
        ax3.set_ylabel('Count')
        ax5.set_ylabel('Count')
        
        handles, labels = ax4.get_legend_handles_labels()
        fig.legend(handles, labels, loc='centre right',bbox_to_anchor=(1, 0.5))
    
    #Convert each data list to numpy arrays
    SP_Peak_ChB1=np.array(SP_Peak_ChB1)
    Ped_Peak_ChB1=np.array(Ped_Peak_ChB1)
    
    SP_Peak_ChD2=np.array(SP_Peak_ChD2)
    Ped_Peak_ChD2=np.array(Ped_Peak_ChD2)
    
    SP_Peak_ChC4=np.array(SP_Peak_ChC4)
    Ped_Peak_ChC4=np.array(Ped_Peak_ChC4)
    
    SP_Peak_ChA3=np.array(SP_Peak_ChA3)
    Ped_Peak_ChA3=np.array(Ped_Peak_ChA3)
    
    SP_Peak_ChA4=np.array(SP_Peak_ChA4)
    Ped_Peak_ChA4=np.array(Ped_Peak_ChA4)
    
    SP_Peak_ChB1A2=np.array(SP_Peak_ChB1A2)
    Ped_Peak_ChB1A2=np.array(Ped_Peak_ChB1A2)
    
    uPedB1=np.array(uPedB1)
    uSPB1=np.array(uSPB1)
    
    uPedD2=np.array(uPedD2)
    uSPD2=np.array(uSPD2)
    
    uPedC4=np.array(uPedC4)
    uSPC4=np.array(uSPC4)
    
    uPedA3=np.array(uPedA3)
    uSPA3=np.array(uSPA3)
    
    uPedA4=np.array(uPedA4)
    uSPA4=np.array(uSPA4)
    
    uPedB1A2=np.array(uPedB1A2)
    uSPB1A2=np.array(uSPB1A2)
    
    
    #Plot 1 p.e. data
    plt.figure()
    plt.scatter(Vb-VBreakDown[2],SP_Peak_ChB1,label='Ch B1',color='red')
    plt.errorbar(Vb-VBreakDown[2],SP_Peak_ChB1,uSPB1,fmt=' ',color='red')
    
    plt.scatter(Vb-VBreakDown[4],SP_Peak_ChD2,label='Ch D2',color='orange')
    plt.errorbar(Vb-VBreakDown[4],SP_Peak_ChD2,uSPD2,fmt=' ',color='orange')
    
    plt.scatter(Vb-VBreakDown[3],SP_Peak_ChC4,label='Ch C4',color='blue')
    plt.errorbar(Vb-VBreakDown[3],SP_Peak_ChC4,uSPC4,fmt=' ',color='blue')
    
    plt.scatter(Vb-VBreakDown[0],SP_Peak_ChA3,label='Ch A3',color='purple')
    plt.errorbar(Vb-VBreakDown[0],SP_Peak_ChA3,uSPA3,fmt=' ',color='purple')
    
    plt.scatter(Vb-VBreakDown[1],SP_Peak_ChA4,label='Ch A4',color='green')
    plt.errorbar(Vb-VBreakDown[1],SP_Peak_ChA4,uSPA4,fmt=' ',color='green')
    
    plt.xlabel("Over Voltage (V)")
    plt.ylabel(XlabelString)
    plt.title("1 p.e. Peak Fit")
    plt.legend()
    
    #Plot pedestal data
    plt.figure()
    plt.scatter(Vb-VBreakDown[2],Ped_Peak_ChB1,label='Ch B1',color='red')
    plt.errorbar(Vb-VBreakDown[2],Ped_Peak_ChB1,uPedA4,fmt=' ',color='red')
    plt.scatter(Vb-VBreakDown[4],Ped_Peak_ChD2,label='Ch D2',color='orange')
    plt.errorbar(Vb-VBreakDown[4],Ped_Peak_ChD2,uPedD2,fmt=' ',color='orange')
    plt.scatter(Vb-VBreakDown[3],Ped_Peak_ChC4,label='Ch C4',color='blue')
    plt.errorbar(Vb-VBreakDown[3],Ped_Peak_ChC4,uPedC4,fmt=' ',color='blue')
    plt.scatter(Vb-VBreakDown[0],Ped_Peak_ChA3,label='Ch A3')
    plt.errorbar(Vb-VBreakDown[0],Ped_Peak_ChA3,uPedA3,fmt=' ')
    plt.scatter(Vb-VBreakDown[1],Ped_Peak_ChA4,label='Ch A4')
    plt.errorbar(Vb-VBreakDown[1],Ped_Peak_ChA4,uPedA4,fmt=' ')
    plt.xlabel("Over Voltage (V)")
    plt.ylabel(XlabelString)
    plt.title("Pedestal Peak Fit")
    plt.legend()
    
    
    #Plot difference between 1 p.e. and pedestal (essentially a better 1 p.e. value)
    uSPmPedB1 = np.sqrt(uSPB1**2+uPedB1**2)
    uSPmPedD2 = np.sqrt(uSPD2**2+uPedD2**2)
    uSPmPedC4 = np.sqrt(uSPC4**2+uPedC4**2)
    uSPmPedA4 = np.sqrt(uSPA4**2+uPedA4**2)
    uSPmPedA3 = np.sqrt(uSPA3**2+uPedA3**2)
    
    SPmPedB1 =  SP_Peak_ChB1-Ped_Peak_ChB1
    SPmPedD2 =  SP_Peak_ChD2-Ped_Peak_ChD2
    SPmPedC4 =  SP_Peak_ChC4-Ped_Peak_ChC4
    SPmPedA3 =  SP_Peak_ChA3-Ped_Peak_ChA3
    SPmPedA4 =  SP_Peak_ChA4-Ped_Peak_ChA4
    SPmPedB1A2 =  SP_Peak_ChB1A2-Ped_Peak_ChB1A2
    
    plt.figure()
    plt.scatter(Vb-VBreakDown[2],SPmPedB1,label='Ch B1',color='red')
    plt.errorbar(Vb-VBreakDown[2],SPmPedB1,uSPmPedB1,fmt=' ',color='red')
    plt.scatter(Vb-VBreakDown[4],SPmPedD2,label='Ch D2',color='orange')
    plt.errorbar(Vb-VBreakDown[4],SPmPedD2,uSPmPedD2,fmt=' ',color='orange')
    plt.scatter(Vb-VBreakDown[3],SPmPedC4,label='Ch C4',color='blue')
    plt.errorbar(Vb-VBreakDown[3],SPmPedC4,uSPmPedC4,fmt=' ',color='blue')
    plt.scatter(Vb-VBreakDown[0],SPmPedA3,label='Ch A3',color='purple')
    plt.errorbar(Vb-VBreakDown[0],SPmPedA3,uSPmPedA3,fmt=' ',color='purple')
    plt.scatter(Vb-VBreakDown[1],SPmPedA4,label='Ch A4',color='green')
    plt.errorbar(Vb-VBreakDown[1],SPmPedA4,uSPmPedA4,fmt=' ',color='green')
    plt.xlabel("Over Voltage (V)")
    plt.ylabel(XlabelString)
    plt.title("1 p.e. - Pedestal Fit")
    plt.legend()
    
    
    
    plt.figure()
    Nph_B1_A = PhotonEfficiencyCalc(CurrentBinsB1A,CurrentNB1A,SPmPedB1,Ped_Peak_ChB1,1,1) 
    Nph_B1_B = PhotonEfficiencyCalc(CurrentBinsB1B,CurrentNB1B,SPmPedB1,Ped_Peak_ChB1,2,1)
    Nph_B1_C = PhotonEfficiencyCalc(CurrentBinsB1C,CurrentNB1C,SPmPedB1,Ped_Peak_ChB1,3,1)
    plt.title('Ch B1')
    plt.xlabel('N')
    plt.ylabel('$P(n_{ph}<=N)$')
    plt.legend()
    
    plt.figure()
    Nph_D2_A = PhotonEfficiencyCalc(CurrentBinsD2A,CurrentND2A,SPmPedD2,Ped_Peak_ChD2,1,3) 
    Nph_D2_B = PhotonEfficiencyCalc(CurrentBinsD2B,CurrentND2B,SPmPedD2,Ped_Peak_ChD2,2,3)
    Nph_D2_C = PhotonEfficiencyCalc(CurrentBinsD2C,CurrentND2C,SPmPedD2,Ped_Peak_ChD2,3,3)
    plt.title('Ch D2')
    plt.xlabel('N')
    plt.ylabel('$P(n_{ph}<=N)$')
    plt.legend()
    
    plt.figure()
    Nph_C4_A = PhotonEfficiencyCalc(CurrentBinsC4A,CurrentNC4A,SPmPedC4,Ped_Peak_ChC4,1,2) 
    Nph_C4_B = PhotonEfficiencyCalc(CurrentBinsC4B,CurrentNC4B,SPmPedC4,Ped_Peak_ChC4,2,2)
    Nph_C4_C = PhotonEfficiencyCalc(CurrentBinsC4C,CurrentNC4C,SPmPedC4,Ped_Peak_ChC4,3,2)
    plt.title('Ch C4')
    plt.xlabel('N')
    plt.ylabel('$P(n_{ph}<=N)$')
    plt.legend()
    
    plt.figure()
    Nph_A3_A = PhotonEfficiencyCalc(CurrentBinsA3A,CurrentNA3A,SPmPedA3,Ped_Peak_ChA3,1,5) 
    Nph_A3_B = PhotonEfficiencyCalc(CurrentBinsA3B,CurrentNA3B,SPmPedA3,Ped_Peak_ChA3,2,5)
    Nph_A3_C = PhotonEfficiencyCalc(CurrentBinsA3C,CurrentNA3C,SPmPedA3,Ped_Peak_ChA3,3,5)
    plt.title('Ch A3')
    plt.xlabel('N')
    plt.ylabel('$P(n_{ph}<=N)$')
    plt.legend()
    
    plt.figure()
    Nph_A4_A = PhotonEfficiencyCalc(CurrentBinsA4A,CurrentNA4A,SPmPedA4,Ped_Peak_ChA4,1,4) 
    Nph_A4_B = PhotonEfficiencyCalc(CurrentBinsA4B,CurrentNA4B,SPmPedA4,Ped_Peak_ChA4,2,4)
    Nph_A4_C = PhotonEfficiencyCalc(CurrentBinsA4C,CurrentNA4C,SPmPedA4,Ped_Peak_ChA4,3,4)
    plt.title('Ch A4')
    plt.xlabel('N')
    plt.ylabel('$P(n_{ph}<=N)$')
    plt.legend()
    
    plt.figure()
    Nph_B1A2_A = PhotonEfficiencyCalc(CurrentBinsB1A2A,CurrentNB1A2A,SPmPedB1A2,Ped_Peak_ChB1A2,1,6) 
    Nph_B1A2_B = PhotonEfficiencyCalc(CurrentBinsB1A2B,CurrentNB1A2B,SPmPedB1A2,Ped_Peak_ChB1A2,2,6)
    Nph_B1A2_C = PhotonEfficiencyCalc(CurrentBinsB1A2C,CurrentNB1A2C,SPmPedB1A2,Ped_Peak_ChB1A2,3,6)
    plt.title('Ch B1-A2')
    plt.xlabel('N')
    plt.ylabel('$P(n_{ph}<=N)$')
    plt.legend()
