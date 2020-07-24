#-*- coding: utf-8 -*-
"""
Created on Wed 08/07/2020

Author: Sam Dekkers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import fftpack
import sys
import glob
from scipy.signal import chirp, find_peaks, peak_widths


#def main(i1, i2, i3, i4, i5, i6):
args = sys.argv
#args = [i1, i2, i3, i4, i5, i6]
#print(len(args))
ArgumentNumber = 7
#ArgumentNumber=6
if(args[1]=='-help'):
    print('Use this file to analyse silicon photomultiplier voltage waveforms - will analyse all files within ~/work/pico/data/analysisfiles, outputing histograms for integrated charge and minimum voltage as well printing to console the average minimum voltage and number of photons for each channel. The arguments should be as follows:')
    print(" ")
    print('python3 DAQanalysis.py [1] [2] [3] [4] [5] [6]')
    print(" ")
    print('[1] = inputstring => use this to distinguish saved histogram filenames')
    print('[2] = number of channels in waveform data')
    print('[3] = waveform input data structure format => 1: [A0, A1, A2 .... D(N-1), D(N)] 2: [A0, B0, C0, D0, A1, B1, .... ,BN, CN, DN]')
    print('[4] = FFT option => 0: No, 1: Yes')
    print('[5] = Print Option => 0: No, 1:Yes')
    print('[6] = String for files to analyse (e.g. for all files analysis_run_1.npy, analysis_run_2.npy,... => input analysis_run')
    print(" ")
    sys.exit()

if(len(args)!=ArgumentNumber):
    print('Wrong no. of arguments! Use:')
    print(" ")
    print('python3 DAQanalysis.py - help')
    print(" ")
    print('for analysis code help.')
    print(" ")
    sys.exit()

inputstring = args[1]
NumberChannels = int(args[2])
if(NumberChannels>4 or NumberChannels < 1):
    print("Invalid Number Of Channels Input!!!!")
    sys.exit()
WaveformFormat = int(args[3])
if(WaveformFormat != 1 and WaveformFormat != 2):
    print("Invalid input format selection!")
    print("Try python3 DAQanalysis.py - help")
    sys.exit()
FFTOn = int(args[4])
if(FFTOn<0 or FFTOn>1):
   print('FFT option invalid!')
   print('Use only Yes (1) or No(0)')
PrintFlag=int(args[5])
AnalysisFilesString="../data/"+args[6]+"*.npy"
#Locate Files
#AnalysisFiles = glob.glob("../data/analysisfiles/*.npy")
AnalysisFiles = glob.glob(AnalysisFilesString)
print(AnalysisFiles)
NumberFiles = len(AnalysisFiles)
print(NumberFiles)

#Allocate Arrays
AllTriggerDepthsA=[]
AllTriggerDepthsB=[]
AllPNA=[]
AllPNB=[]
StoreIntegrationsA=[]
StoreIntegrationsB=[]
MinVoltageA=[]
MinVoltageB=[]
AllTriggerDepthsC=[]
AllTriggerDepthsD=[]
AllPNC=[]
AllPND=[]
StoreIntegrationsC=[]
StoreIntegrationsD=[]
MinVoltageC=[]
MinVoltageD=[]


#All Data Averages
ChAAve=[]
ChBAve=[]
ChCAve=[]
ChDAve=[]

#Integrals of each channel
IntA=[]
IntASig=[]
IntB=[]
IntC=[]
IntD=[]
IntBSig=[]
IntCSig=[]
IntDSig=[]

#Channel signal counters
CounterA = 0
CounterB = 0
CounterC = 0
CounterD = 0
CounterAp = 0
CounterBp = 0
CounterCp = 0
CounterDp = 0
#Peak Indices
StartIndices = []
StopIndices = []
#Set parameters
Samples = 508 #512 for pre cumsum
dt = 0.8e-9
TimeArray = np.linspace(0.0,Samples*dt,Samples)
fftT = np.linspace(0.0, 1.0/(2.0*dt), Samples//2)
PhotonCalibration = 25.0*2e-9 #two SiPM vs intial tests (~25 integrated charge 1 p,e,)
ChannelPolarity = [-1,1,-1,-1]
SignalCounter=0
NoiseCounter=0
TotalEvents = 0
#Analyse each file
for j in range(NumberFiles):
    File = AnalysisFiles[j]
    filedata = np.array(np.load(File,allow_pickle=True))
    rawdata=filedata[6]
    print("Reading file: ",File) 
    #print(data.shape)
    #print(File)
    #print(data)
    #print(len(data[1]))
    #rawdata = data[0]
    #movave = np.cumsum(np.insert(rawdata,0,0))
    #movave[5:]=movave[5:]-movave[:-5]
    #data = (movave[5:]-movave[:-5])/float(5)
    #print(data)
 #figcheck = plt.figure()
# plt.subplot(1,2,1)
#    plt.plot(rawdata)
#    plt.subplot(1,2,2)
#    plt.plot(data)
#    plt.show()

    #Allocate File Arrays/Parameters
    NumberTriggers = len(rawdata)
    TotalEvents+=NumberTriggers
    TriggerDepthA = []
    TriggerDepthB = []
    PhotonNumberA = []
    PhotonNumberB = []
    TriggerDepthC = []
    TriggerDepthD = []
    PhotonNumberC = []
    PhotonNumberD = []
    print("Recorded Waveforms = ",NumberTriggers)
    WaveformsPerChannel = NumberTriggers/NumberChannels
    print("Number of waveforms per channel = ",WaveformsPerChannel)
    #ChannelCounter = 1
    #print(rawdata.shape)
    #print(rawdata)

    ChannelCounter=0
    NoiseThresholdAmp = 10
    NoiseThreshold = 2

    #Determine average channel value before analysis
    NoiseA=0
    NoiseB=0
    NoiseC=0
    NoiseD=0
    for i in range(NumberTriggers):
         data=rawdata[i]
         noiselevel=np.average(data)
         ChannelCounter+=1
         if(ChannelCounter>NumberChannels):ChannelCounter=1
         if(ChannelCounter==1): NoiseA+=noiselevel
         if(ChannelCounter==2): NoiseB+=noiselevel
         if(ChannelCounter==3): NoiseC+=noiselevel
         if(ChannelCounter==4): NoiseD+=noiselevel


    ThreshA = NoiseA/NumberTriggers
    ThreshB = NoiseB/NumberTriggers
    ThreshC = NoiseC/NumberTriggers
    ThreshD = NoiseD/NumberTriggers
    print("Channel A Threshold = ",ThreshA)
    print("Channel B Threshold = ",ThreshB)
    print("Channel C Threshold = ",ThreshC)
    print("Channel D Threshold = ",ThreshD)
    ChannelCounter=0
    StartIndex=0
    EndIndex=0
    IntegralThresh = 1e-7
    #Analyse each waveform in the file
    for i in range(NumberTriggers):
         #Find Pulse
         moveavefilt=np.cumsum(np.insert(rawdata[i],0,0))
         data=(moveavefilt[5:]-moveavefilt[:-5])/float(5)
         ChannelCounter+=1
         #if(ChannelCounter==2):
         #   plt.plot(data)
         #   plt.show()
         #if(ChannelCounter==4): ChannelCounter=0
         MoveAveN = 5
         AveCounter = 0
         NoAveCounter = 0
         PulseFlag = 0
         ThreshAve = 0.9

         if(ChannelCounter==1):ThreshAve = abs(ThreshA)
         elif(ChannelCounter==2):ThreshAve = abs(ThreshB)
         elif(ChannelCounter==3):ThreshAve = abs(ThreshC)
         elif(ChannelCounter==4):ThreshAve = abs(ThreshD)
         yes = 1
         if(yes==1):
          # dV = diff(data)
          # dT = difff(TimeArray)
          # dVdT=float(dV/dT)
           IntegrateWaveform = simps(data,TimeArray)
           #print("Integral: ",IntegrateWaveform)
           if(ChannelCounter==1): 
                IntA.append(IntegrateWaveform)
                IntASig.append(IntegrateWaveform)
                fl = 0
                if(IntegrateWaveform>IntegralThresh):
                    CounterA+=1
                    fl=1
                #else:
                   # plt.plot(data)
                   # plt.title("Int not above")
                   # plt.show()
                if(np.min(data)<-10.0):
                    CounterAp+=1
                elif(fl==1):
                    plt.plot(data)
                    plt.title("Int above thresh but Peak Not")
                    #plt.show()
                    
           elif(ChannelCounter==2):
                IntB.append(IntegrateWaveform)
                IntBSig.append(IntegrateWaveform)
                if(IntegrateWaveform>IntegralThresh): CounterB+=1
                if(np.max(data)>10.0): CounterBp+=1
           elif(ChannelCounter==3):
                IntC.append(IntegrateWaveform)
                IntCSig.append(IntegrateWaveform)
                if(IntegrateWaveform>IntegralThresh): CounterC+=1
                if(np.min(data)<-6.0): CounterCp+=1
           elif(ChannelCounter==4):
                IntD.append(IntegrateWaveform)
                IntDSig.append(IntegrateWaveform)
                if(IntegrateWaveform>IntegralThresh): CounterD+=1
                if(np.min(data)<-6.0): CounterDp+=1
           
           #peaks,_=find_peaks(data)
           #results_half = peak_widths(data, peaks, rel_height=0.5)
           #results_full = peak_widths(data, peaks, rel_height=1)
           #print("Max Peak: ",np.max(data[peaks]))
           #print("Min Peak: ",np.min(data[peaks]))
           #print("Max Width: ",0.8*np.max(results_full))
           #print("Max Half  Width: ", 0.8*np.max(results_half)) 
           #plt.plot(data)
           #plt.plot(peaks,data[peaks])
           #plt.hlines(*results_half[1:], color="C2")
           #plt.hlines(*results_full[1:], color="C3")           
          # plt.show()
           
 
         if(ChannelPolarity[ChannelCounter-1]==-1):
             for k in range(len(data)-MoveAveN):
                 MoveAverage = (data)[k:k+MoveAveN]
                 MoveAverageVal =  np.average(MoveAverage)
             	 #print("Index: ",k)
             	 #print("5 point Ave = ",MoveAverageVal) 
             
                 if(PulseFlag==0 and MoveAverageVal<=-ThreshAve): AveCounter+=1
                 else: AveCounter=0

                 if(AveCounter>5):
                     StartIndex = k-5
                     PulseFlag = 1
                 if(PulseFlag==1 and MoveAverageVal>-ThreshAve): NoAveCounter+=1 
                 else: NoAveCounter = 0
       
                 if(NoAveCounter>5):
                     EndIndex = k+5
                     if(np.min((data)[StartIndex:EndIndex])!=np.min(data)):
                        PulseFlag=0
                        AveCounter=0
                        NoAveCounter=0
                     else:
                        PulseFlag = 0
                        break
                #print("AveCount", AveCounter)
                #print("NoAveCount", NoAveCounter)
         if(ChannelPolarity[ChannelCounter-1]==1):
             for k in range(len(data)-MoveAveN):
                 MoveAverage = (data)[k:k+MoveAveN]
                 MoveAverageVal =  np.average(MoveAverage)
             	 #print("Index: ",k)
             	 #print("5 point Ave = ",MoveAverageVal) 
                 if(PulseFlag==0 and MoveAverageVal>=ThreshAve):AveCounter+=1
                 else:AveCounter=0

                 if(AveCounter>5):
                     StartIndex = k-5
                     PulseFlag = 1
                 if(PulseFlag==1 and MoveAverageVal<ThreshAve): NoAveCounter+=1 
                 else: NoAveCounter = 0
       
                 if(NoAveCounter>5):
                     EndIndex = k+5
                     if(np.max((data)[StartIndex:EndIndex])!=np.max(data)):
                        PulseFlag=0
                        AveCounter=0
                        NoAveCounter=0
                     else:
                        PulseFlag = 0
                        break
                 #print("AveCount", AveCounter)
                 #print("NoAveCount", NoAveCounter)
        
         if(PrintFlag==1): 
             print("################################") 
             print("Found start index = ", StartIndex)
             print("Found end index = ", EndIndex)       
        
         StopIndices.append(EndIndex)
         StartIndices.append(StartIndex)


         if(FFTOn==1):
            fftV = fftpack.fft(data)
            plt.plot(fftT, 2.0/Samples * np.abs(fftV[0:Samples//2]))
            plt.grid()
            plt.show()
            samplefreq=fftpack.fftfreq(Samples, dt)
            Copy = fftV.copy()
            Copy[np.abs(samplefreq)>0.15e8]=0
            filteredsig=fftpack.ifft(Copy)
            nonfiltered=fftpack.ifft(fftV)
            plt.plot(filteredsig,color='red')
            plt.plot(nonfiltered,color='blue')
            plt.grid()
            plt.show()


         #plt.plot((data[0])[StartIndex:EndIndex])
         #plt.show()
         #Numerical Integration using Simpsons Rule 
         if(StartIndex>=EndIndex or StartIndex==0 or EndIndex==0): 
             Area = 0.0
             NoiseCounter+=1
         else: Area = -simps((data)[StartIndex:EndIndex],TimeArray[StartIndex:EndIndex])
         PN = int(Area/PhotonCalibration)
         
         if(PrintFlag==1):
             print("Intergrated Pulse: ",Area)
             print("Min Voltage = ",np.min(data))
             print("Number of photons = ", PN)
         
         #if(PN<10):
         #  plt.plot((data[i])[StartIndex:EndIndex])
         #  plt.show() 
         if(WaveformFormat==1):
            if(i<WaveformsPerChannel):
               TriggerDepthA.append(np.min(data))
               PhotonNumberA.append(PN)
               StoreIntegrationsA.append(Area)
               MinVoltageA.append(np.min(data))

            elif(i>=WaveformsPerChannel and i<2*WaveformsPerChannel):
               TriggerDepthB.append(np.min(data))
               PhotonNumberB.append(PN)
               StoreIntegrationsB.append(Area)
               MinVoltageB.append(np.min(data))

            elif(i>=2*WaveformsPerChannel and i<3*WaveformsPerChannel):
               TriggerDepthC.append(np.min(data))
               PhotonNumberC.append(PN)
               StoreIntegrationsC.append(Area)
               MinVoltageC.append(np.min(data))
         
            elif(i>=3*WaveformsPerChannel and i<4*WaveformsPerChannel):
               TriggerDepthD.append(np.min(data))
               PhotonNumberD.append(PN)
               StoreIntegrationsD.append(Area)
               MinVoltageD.append(np.min(data))
         
         elif(WaveformFormat==2): 
            if(ChannelCounter==1):
               TriggerDepthA.append(np.min(data))
               PhotonNumberA.append(PN)
               StoreIntegrationsA.append(Area)
               if(np.min(data)<=-NoiseThresholdAmp): MinVoltageA.append(np.min(data))
               ChAAve.append(np.average(data))
               #if(NumberChannels>1): ChannelCounter=2

            elif(ChannelCounter==2):
               TriggerDepthB.append(np.min(data))
               PhotonNumberB.append(PN)
               StoreIntegrationsB.append(Area)
               #MinVoltageB.append(np.min(data))
               if(np.max(data)>=NoiseThresholdAmp): MinVoltageB.append(np.max(data))
               ChAAve.append(np.average(data))
               ChBAve.append(np.average(data))
              # if(NumberChannels==2): ChannelCounter=1
              # elif(NumberChannels>2): ChannelCounter=3

            elif(ChannelCounter==3):
               TriggerDepthC.append(np.min(data))
               PhotonNumberC.append(PN)
               StoreIntegrationsC.append(Area)
               #MinVoltageC.append(np.min(data))
               if(np.min(data)<=-NoiseThreshold): MinVoltageC.append(np.min(data))
               ChAAve.append(np.average(data))
               ChCAve.append(np.average(data))
              # if(NumberChannels==3): ChannelCounter=1
              # elif(NumberChannels==4): ChannelCounter=4
         
            elif(ChannelCounter==4):
               TriggerDepthD.append(np.min(data))
               PhotonNumberD.append(PN)
               StoreIntegrationsD.append(Area)
               #MinVoltageD.append(np.min(data))
               if(np.min(data)<=-NoiseThreshold): MinVoltageD.append(np.min(data))
               ChAAve.append(np.average(data))
              # ChannelCounter=1
               ChDAve.append(np.average(data))
       
         if(PrintFlag==1): print("################################")
    
         if(ChannelCounter==NumberChannels):ChannelCounter=0
 

    AverageTriggerDepthA = np.average(TriggerDepthA)
    AveragePNA = np.average(PhotonNumberA)
    AllTriggerDepthsA.append(AverageTriggerDepthA)
    AllPNA.append(AveragePNA)
    if(NumberChannels>=2):    
       AverageTriggerDepthB = np.average(TriggerDepthB)
       AveragePNB = np.average(PhotonNumberB)
       AllTriggerDepthsB.append(AverageTriggerDepthB)
       AllPNB.append(AveragePNB)
    
    if(NumberChannels>2):    
       AverageTriggerDepthC = np.average(TriggerDepthC)
       AveragePNC = np.average(PhotonNumberC)
       AllTriggerDepthsC.append(AverageTriggerDepthC)
       AllPNC.append(AveragePNC)
    
    if(NumberChannels==4):    
       AverageTriggerDepthD = np.average(TriggerDepthD)
       AveragePND = np.average(PhotonNumberD)
       AllTriggerDepthsD.append(AverageTriggerDepthD)
       AllPND.append(AveragePND)
    

AllFileAverageDepthA = np.average(AllTriggerDepthsA)
AllFileAveragePNA = np.average(AllPNA)
print("Average minimum voltage in Channel A = %.2f mV" %(AllFileAverageDepthA))
print("Average Photon Number A = ",AllFileAveragePNA)
if(NumberChannels>=2):
    AllFileAverageDepthB = np.average(AllTriggerDepthsB)
    AllFileAveragePNB = np.average(AllPNB)
    print("Average minimum voltage in Channel B = %.2f mV" %(AllFileAverageDepthB))
    print("Average Photon Number B = ",AllFileAveragePNB)
if(NumberChannels>2):
    AllFileAverageDepthC = np.average(AllTriggerDepthsB)
    AllFileAveragePNC = np.average(AllPNB)
    print("Average minimum voltage in Channel C = %.2f mV" %(AllFileAverageDepthC))
    print("Average Photon Number C = ",AllFileAveragePNC)
if(NumberChannels==4):
    AllFileAverageDepthD = np.average(AllTriggerDepthsB)
    AllFileAveragePND = np.average(AllPNB)
    print("Average minimum voltage in Channel D = %.2f mV" %(AllFileAverageDepthD))
    print("Average Photon Number D = ",AllFileAveragePND)

print("Noise count: %d / %d " %(NoiseCounter,TotalEvents))
NumBins = 100

print("Ch A Int Count: ", CounterA)
print("Ch A Peak Count: ",CounterAp)
print("Ch B Int Count: ", CounterB)
print("Ch B Peak Count: ",CounterBp)
print("Ch C Int Count: ", CounterC)
print("Ch C Peak Count: ",CounterCp)
print("Ch D Int Count: ", CounterD)
print("Ch D Peak Count: ",CounterDp)


#Histograms of integrated pulses
fig = plt.figure(1)

if(NumberChannels==2):
    plt.subplot(1,2,1)
elif(NumberChannels>2):
    plt.subplot(2,2,1)

plt.hist(StoreIntegrationsA, bins=NumBins)
plt.title("Integrated Charge A")
plt.xlabel("Integrated Charge")
plt.ylabel("Number")

if(NumberChannels==1):
   plt.show()
   savestring = 'IntegratedHist_'+inputstring+'.pdf'
   plt.savefig(savestring, bbox_inches='tight')

if(NumberChannels==2):
    plt.subplot(1,2,2)
elif(NumberChannels>2):
    plt.subplot(2,2,2)

if(NumberChannels>=2):
    plt.hist(StoreIntegrationsB, bins = NumBins)
    plt.title("Integrated Charge B")
    plt.xlabel("Integrated Charge")
    plt.ylabel("Number")

if(NumberChannels==2):
    plt.show()
    savestring = 'IntegratedHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')


if(NumberChannels>2):
    plt.subplot(2,2,3)
    plt.hist(StoreIntegrationsC, bins=NumBins)
    plt.title("Integrated Charge C")
    plt.xlabel("Integrated Charge")
    plt.ylabel("Number")
if(NumberChannels==3):
    plt.show()
    savestring = 'IntegratedHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')

if(NumberChannels==4):
    plt.subplot(2,2,4)
    plt.hist(StoreIntegrationsD, bins=NumBins)
    plt.title("Integrated Charge D")
    plt.xlabel("Integrated Charge")
    plt.ylabel("Number")
    plt.show()
    savestring = 'IntegratedHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')

#Voltage Histograms
figV = plt.figure(2)

if(NumberChannels==2):
    plt.subplot(1,2,1)
elif(NumberChannels>2):
    plt.subplot(2,2,1)

nA,vA, _ = plt.hist(MinVoltageA,bins=NumBins)
plt.title("Peak Voltage Channel A")
plt.xlabel("Peak Voltage (mV)")
plt.ylabel("Number")
#print(vA)
#nA = np.sort(nA)
nAi = np.argsort(nA)
nAsize = len(nA)
print("Peak Channel A [%f, %f]" %(vA[nAi[nAsize-2]],nA[nAi[nAsize-2]]))
if(NumberChannels==1):
   plt.show()
   savestring = 'VoltageHist_'+inputstring+'.pdf'
   plt.savefig(savestring, bbox_inches='tight')

if(NumberChannels==2):
    plt.subplot(1,2,2)
elif(NumberChannels>2):
    plt.subplot(2,2,2)

if(NumberChannels>=2):
    nB, vB, _ = plt.hist(MinVoltageB, bins = NumBins)
    plt.title("Peak Voltage B")
    plt.xlabel("Peak Voltage (mV)")
    plt.ylabel("Number")
    nBi = np.argsort(nB)
    nBsize = len(nB)
    print("Peak Channel B [%f, %f]" %(vB[nBi[nBsize-1]],nB[nBi[nBsize-1]]))

if(NumberChannels==2):
    plt.show()
    savestring = 'VoltageHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')


if(NumberChannels>2):
    plt.subplot(2,2,3)
    nC, vC, _ = plt.hist(MinVoltageC, bins=NumBins)
    plt.title("Peak Voltage C")
    plt.xlabel("Peak Voltage (mV)")
    plt.ylabel("Number")
    nCi = np.argsort(nC)
    nCsize = len(nC)
    print("Peak Channel C [%f, %f]" %(vC[nCi[nCsize-1]],nC[nCi[nCsize-1]]))
if(NumberChannels==3):
    plt.show()
    savestring = 'VoltageHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')

if(NumberChannels==4):
    plt.subplot(2,2,4)
    nD, vD, _ = plt.hist(MinVoltageD, bins=NumBins)
    plt.title("Peak Voltage D")
    plt.xlabel("Peak Voltage (mV)")
    plt.ylabel("Number") 
    nDi = np.argsort(nD)
    nDsize = len(nD)
    print("Peak Channel D [%f, %f]" %(vD[nDi[nDsize-1]],nD[nDi[nDsize-1]]))
    plt.show()
    savestring = 'VoltageHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')

#Histograms of average voltage
figAv = plt.figure(3)

if(NumberChannels==2):
    plt.subplot(1,2,1)
elif(NumberChannels>2):
    plt.subplot(2,2,1)

plt.hist(ChAAve, bins=NumBins)
plt.title("Average Voltage A")
plt.xlabel("Voltage (mV)")
plt.ylabel("Number")

if(NumberChannels==1):
   plt.show()
   savestring = 'AvVoltHist_'+inputstring+'.pdf'
   plt.savefig(savestring, bbox_inches='tight')

if(NumberChannels==2):
    plt.subplot(1,2,2)
elif(NumberChannels>2):
    plt.subplot(2,2,2)

if(NumberChannels>=2):
    plt.hist(ChBAve, bins = NumBins)
    plt.title("Average Voltage B")
    plt.xlabel("Voltage (mV)")
    plt.ylabel("Number")

if(NumberChannels==2):
    plt.show()
    savestring = 'AvVoltHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')


if(NumberChannels>2):
    plt.subplot(2,2,3)
    plt.hist(ChCAve, bins=NumBins)
    plt.title("Average Voltage C")
    plt.xlabel("Voltage (mV)")
    plt.ylabel("Number")
if(NumberChannels==3):
    plt.show()
    savestring = 'AveVoltHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')

if(NumberChannels==4):
    plt.subplot(2,2,4)
    plt.hist(ChDAve, bins=NumBins)
    plt.title("Average Voltage D")
    plt.xlabel("Voltage (mV)")
    plt.ylabel("Number")
    plt.show()
    savestring = 'IntegratedHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')

#Plot Entire Waveform Integral
figInt = plt.figure(4) 
plt.subplot(2,2,1)
plt.hist(IntA, bins = NumBins)
plt.title("Integral of Waveform Ch A")
plt.xlabel("Integral (mV s)")
plt.ylabel("Number")
plt.show()

plt.subplot(2,2,2)
plt.hist(IntB, bins = NumBins)
plt.title("Integral of Waveform Ch B")
plt.xlabel("Integral (mV s)")
plt.ylabel("Number")
plt.show()

plt.subplot(2,2,3)
plt.hist(IntC, bins = NumBins)
plt.title("Integral of Waveform Ch C")
plt.xlabel("Integral (mV s)")
plt.ylabel("Number")
plt.show()

plt.subplot(2,2,4)
plt.hist(IntD, bins = NumBins)
plt.title("Integral of Waveform Ch D")
plt.xlabel("Integral (mV s)")
plt.ylabel("Number")
plt.show()

figStart = plt.figure(5) 
plt.hist(StartIndices, bins = NumBins)
plt.title("Peak Start Indices All Channels")
plt.xlabel("Start  Index")
plt.ylabel("Number")
plt.show()
figStop = plt.figure(6) 
plt.hist(StopIndices, bins = NumBins)
plt.title("Peak Stop Indices All Channels")
plt.xlabel("Stop  Index")
plt.ylabel("Number")
plt.show()
