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

args = sys.argv
#print(len(args))
ArgumentNumber = 5

if(args[1]=='-help'):
    print('Use this file to analyse silicon photomultiplier voltage waveforms - will analyse all files within ~/work/pico/data/analysisfiles, outputing histograms for integrated charge and minimum voltage as well printing to console the average minimum voltage and number of photons for each channel. The arguments should be as follows:')
    print(" ")
    print('python3 DAQanalysis.py [1] [2] [3] [4]')
    print(" ")
    print('[1] = inputstring => use this to distinguish saved histogram filenames')
    print('[2] = number of channels in waveform data')
    print('[3] = waveform input data structure format => 1: [A0, A1, A2 .... D(N-1), D(N)] 2: [A0, B0, C0, D0, A1, B1, .... ,BN, CN, DN]')
    print('[4] = FFT option => 0: No, 1: Yes')
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

#Locate Files
AnalysisFiles = glob.glob("../data/analysisfiles/*.npy")
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

#Set parameters
Samples = 508 #512 for pre cumsum
dt = 0.8e-9
TimeArray = np.linspace(0.0,Samples*dt,Samples)
fftT = np.linspace(0.0, 1.0/(2.0*dt), Samples//2)
PhotonCalibration = 25.0*2e-9 #two SiPM vs intial tests (~25 integrated charge 1 p,e,)

#Analyse each file
for j in range(NumberFiles):
    File = AnalysisFiles[j]
    rawdata = np.array(np.load(File))
    #print(data.shape)
    #print(File)
    #print(data)
    #print(len(data[1]))
    #rawdata = data[0]
    #movave = np.cumsum(np.insert(rawdata,0,0))
    #movave[5:]=movave[5:]-movave[:-5]
    #data = (movave[5:]-movave[:-5])/float(5)
    #print(data)
#    figcheck = plt.figure()
#    plt.subplot(1,2,1)
#    plt.plot(rawdata)
#    plt.subplot(1,2,2)
#    plt.plot(data)
#    plt.show()

    #Allocate File Arrays/Parameters
    NumberTriggers = len(rawdata)
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
    ChannelCounter = 1

    #Analyse each waveform in the file
    for i in range(NumberTriggers):
         #Find Pulse
         moveavefilt=np.cumsum(np.insert(rawdata[i],0,0))
         data=(moveavefilt[5:]-moveavefilt[:-5])/float(5)
         MoveAveN = 5
         AveCounter = 0
         NoAveCounter = 0
         PulseFlag = 0
         ThreshAve = -0.9
         for k in range(len(data)-MoveAveN):
             MoveAverage = (data)[k:k+MoveAveN]
             MoveAverageVal =  np.average(MoveAverage)
             #print("Index: ",k)
             #print("5 point Ave = ",MoveAverageVal) 
             if(PulseFlag==0 and MoveAverageVal<=ThreshAve): AveCounter+=1
             else: AveCounter=0

             if(AveCounter>5):
                 StartIndex = k-5
                 PulseFlag = 1
             if(PulseFlag==1 and MoveAverageVal>ThreshAve): NoAveCounter+=1 
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
         print("################################") 
         print("Found start index = ", StartIndex)
         print("Found end index = ", EndIndex)       
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
         Area = -simps((data)[StartIndex:EndIndex],TimeArray[StartIndex:EndIndex])
         print("Intergrated Pulse: ",Area)
         print("Min Voltage = ",np.min(data))
         PN = int(Area/PhotonCalibration)
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
               MinVoltageA.append(np.min(data))
               if(NumberChannels>1): ChannelCounter=2

            elif(ChannelCounter==2):
               TriggerDepthB.append(np.min(data))
               PhotonNumberB.append(PN)
               StoreIntegrationsB.append(Area)
               MinVoltageB.append(np.min(data))
               if(NumberChannels==2): ChannelCounter=1
               elif(NumberChannels>2): ChannelCounter=3

            elif(ChannelCounter==3):
               TriggerDepthC.append(np.min(data))
               PhotonNumberC.append(PN)
               StoreIntegrationsC.append(Area)
               MinVoltageC.append(np.min(data))
               if(NumberChannels==3): ChannelCounter=1
               elif(NumberChannels==4): ChannelCounter=4
         
            elif(ChannelCounter==4):
               TriggerDepthD.append(np.min(data))
               PhotonNumberD.append(PN)
               StoreIntegrationsD.append(Area)
               MinVoltageD.append(np.min(data))
               ChannelCounter=1
         print("################################")
    
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

#Histograms of integrated pulses
fig = plt.figure()

if(NumberChannels==2):
    plt.subplot(1,2,1)
elif(NumberChannels>2):
    plt.subplot(2,2,1)

plt.hist(StoreIntegrationsA, bins=50)
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
    plt.hist(StoreIntegrationsB, bins = 50)
    plt.title("Integrated Charge B")
    plt.xlabel("Integrated Charge")
    plt.ylabel("Number")

if(NumberChannels==2):
    plt.show()
    savestring = 'IntegratedHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')


if(NumberChannels>2):
    plt.subplot(2,2,3)
    plt.hist(StoreIntegrationsC, bins=50)
    plt.title("Integrated Charge C")
    plt.xlabel("Integrated Charge")
    plt.ylabel("Number")
if(NumberChannels==3):
    plt.show()
    savestring = 'IntegratedHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')

if(NumberChannels==4):
    plt.subplot(2,2,4)
    plt.hist(StoreIntegrationsD, bins=50)
    plt.title("Integrated Charge D")
    plt.xlabel("Integrated Charge")
    plt.ylabel("Number")
    plt.show()
    savestring = 'IntegratedHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')

#Voltage Histograms
figV = plt.figure()

if(NumberChannels==2):
    plt.subplot(1,2,1)
elif(NumberChannels>2):
    plt.subplot(2,2,1)

plt.hist(MinVoltageA,bins=50)
plt.title("Peak Voltage Channel A")
plt.xlabel("Peak Voltage (mV)")
plt.ylabel("Number")

if(NumberChannels==1):
   plt.show()
   savestring = 'VoltageHist_'+inputstring+'.pdf'
   plt.savefig(savestring, bbox_inches='tight')

if(NumberChannels==2):
    plt.subplot(1,2,2)
elif(NumberChannels>2):
    plt.subplot(2,2,2)

if(NumberChannels>=2):
    plt.hist(MinVoltageB, bins = 50)
    plt.title("Peak Voltage B")
    plt.xlabel("Peak Voltage (mV)")
    plt.ylabel("Number")

if(NumberChannels==2):
    plt.show()
    savestring = 'VoltageHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')


if(NumberChannels>2):
    plt.subplot(2,2,3)
    plt.hist(MinVoltageC, bins=50)
    plt.title("Peak Voltage C")
    plt.xlabel("Peak Voltage (mV)")
    plt.ylabel("Number")
if(NumberChannels==3):
    plt.show()
    savestring = 'VoltageHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')

if(NumberChannels==4):
    plt.subplot(2,2,4)
    plt.hist(MinVoltageD, bins=50)
    plt.title("Peak Voltage D")
    plt.xlabel("Peak Voltage (mV)")
    plt.ylabel("Number")
    plt.show()
    savestring = 'VoltageHist_'+inputstring+'.pdf'
    plt.savefig(savestring, bbox_inches='tight')
