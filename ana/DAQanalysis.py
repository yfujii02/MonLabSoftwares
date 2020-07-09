#-*- coding: utf-8 -*-
"""
Created on Wed 08/07/2020

Author: Sam Dekkers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import sys
import glob

args = sys.argv
#print(len(args))
ArgumentNumber = 3

if(args[1]=='-help'):
    print('Use this file to analyse silicon photomultiplier voltage waveforms - will analyse all files within ~/work/pico/data/analysisfiles, outputing histograms for integrated charge and minimum voltage as well printing to console the average minimum voltage and number of photons for each channel. The arguments should be as follows:')
    print(" ")
    print('python3 DAQanalysis.py [1] [2]')
    print(" ")
    print('[1] = inputstring => use this to distinguish saved histogram filenames')
    print('[2] = number of channels in waveform data')
    print(" ")
    sys.exit()

if(len(args)!=ArgumentNumber):
    print('Wrong no. of arguments! Use:')
    print(" ")
    print('python3 DAQanalysis - help')
    print(" ")
    print('for analysis code help.')
    print(" ")
    sys.exit()

inputstring = args[1]
NumberChannels = int(args[2])


AnalysisFiles = glob.glob("../data/analysisfiles/*.npy")
print(AnalysisFiles)
NumberFiles = len(AnalysisFiles)
print(NumberFiles)
AllTriggerDepthsA=[]
AllTriggerDepthsB=[]
AllPNA=[]
AllPNB=[]
StoreIntegrationsA=[]
StoreIntegrationsB=[]
MinVoltageA=[]
MinVoltageB=[]
Samples = 512
dt = 0.8
TimeArray = np.linspace(0,Samples*dt,Samples)
PhotonCalibration = 25.0*2 #two SiPM vs intial tests (~25 integrated charge 1 p,e,)
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

    NumberTriggers = len(rawdata)
    TriggerDepthA = []
    TriggerDepthB = []
    PhotonNumberA = []
    PhotonNumberB = []
    print("Recorded Waveforms = ",NumberTriggers)
    WaveformsPerChannel = NumberTriggers/NumberChannels
    print("Number of waveforms per channel = ",WaveformsPerChannel)

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

         else:
            #SET UP CHANNEL C AND D HERE 
            print("analyis code incomplete")

         print("################################")
    
    AverageTriggerDepthA = np.average(TriggerDepthA)
    AverageTriggerDepthB = np.average(TriggerDepthB)
    AveragePNA = np.average(PhotonNumberA)
    AveragePNB = np.average(PhotonNumberB)
   # print(AverageTriggerDepth)
    AllTriggerDepthsA.append(AverageTriggerDepthA)
    AllTriggerDepthsB.append(AverageTriggerDepthB)
    AllPNA.append(AveragePNA)
    AllPNB.append(AveragePNB)

AllFileAverageDepthA = np.average(AllTriggerDepthsA)
AllFileAverageDepthB = np.average(AllTriggerDepthsB)
AllFileAveragePNA = np.average(AllPNA)
AllFileAveragePNB = np.average(AllPNB)
print("Average minimum voltage in Channel A = %.2f mV" %(AllFileAverageDepthA))
print("Average minimum voltage in Channel B = %.2f mV" %(AllFileAverageDepthB))
print("Average Photon Number A = ",AllFileAveragePNA)
print("Average Photon Number B = ",AllFileAveragePNB)
#Histograms of integrated pulses
fig = plt.figure()
binsA = int((np.max(StoreIntegrationsA)-np.min(StoreIntegrationsA))/np.sqrt(len(StoreIntegrationsA)))
print(binsA)
plt.subplot(1,2,1)
plt.hist(StoreIntegrationsA, bins=2*binsA)
plt.title("Integrated Charge A")
plt.xlabel("Integrated Charge")
plt.ylabel("Number")

binsB = int((np.max(StoreIntegrationsB)-np.min(StoreIntegrationsB))/np.sqrt(len(StoreIntegrationsB)))
print(binsB)
plt.subplot(1,2,2)
plt.hist(StoreIntegrationsB, bins = 2*binsB)
plt.title("Integrated Charge B")
plt.xlabel("Integrated Charge")
plt.ylabel("Number")
savestring = 'IntegratedHist_'+inputstring+'.pdf'
plt.savefig(savestring, bbox_inches='tight')
plt.show()

figV = plt.figure()
binsVA = int(abs(np.max(AllTriggerDepthsA)-np.min(AllTriggerDepthsA))/np.sqrt(len(AllTriggerDepthsA)))
print(binsVA)
plt.subplot(1,2,1)
plt.hist(AllTriggerDepthsA,bins=binsA)
plt.title("Peak Voltage Channel A")
plt.xlabel("Peak Voltage (mV)")
plt.ylabel("Number")

binsVB = int(abs(np.max(AllTriggerDepthsB)-np.min(AllTriggerDepthsB))/np.sqrt(len(AllTriggerDepthsB)))
print(binsVB)
plt.subplot(1,2,2)
plt.hist(AllTriggerDepthsB,bins=binsB)
plt.title("Peak Voltage Channel B")
plt.xlabel("Peak Voltage (mV)")
plt.ylabel("Number")
savestringV = 'VoltageHist_'+inputstring+'.pdf' 
plt.savefig(savestringV,bbox_inches='tight')
plt.show()







