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

AnalysisFiles = glob.glob("../data/analysisfiles/*.npy")
print(AnalysisFiles)
NumberFiles = len(AnalysisFiles)
print(NumberFiles)
AllTriggerDepthsA=[]
AllTriggerDepthsB=[]
AllPNA=[]
AllPNB=[]
Samples = 512
dt = 0.8
TimeArray = np.linspace(0,Samples*dt,Samples)
PhotonCalibration = 20.0
for j in range(NumberFiles):
    File = AnalysisFiles[j]
    data = np.array(np.load(File))
    print(data.shape)
    print(File)
    print(data)
    print(len(data[1]))
    NumberTriggers = len(data)
    TriggerDepthA = []
    TriggerDepthB = []
    PhotonNumberA = []
    PhotonNumberB = []
    print(NumberTriggers)

    for i in range(NumberTriggers):
         #Find Pulse
         MoveAveN = 5
         AveCounter = 0
         NoAveCounter = 0
         PulseFlag = 0
         ThreshAve = -0.9
         for k in range(len(data[0])-MoveAveN):
             MoveAverage = (data[0])[k:k+MoveAveN]
             MoveAverageVal =  np.average(MoveAverage)
             print("Index: ",k)
             print("5 point Ave = ",MoveAverageVal) 
             if(PulseFlag==0 and MoveAverageVal<=ThreshAve): AveCounter+=1
             else: AveCounter=0

             if(AveCounter>5):
                 StartIndex = k-3
                 PulseFlag = 1
             if(PulseFlag==1 and MoveAverageVal>ThreshAve): NoAveCounter+=1 
             else: NoAveCounter = 0
       
             if(NoAveCounter>5):
                 EndIndex = k-3
                 PulseFlag = 0
                 break
             print("AveCount", AveCounter)
             print("NoAveCount", NoAveCounter)
    
         print("Found start index = ", StartIndex)
         print("Found end index = ", EndIndex)       
         #plt.plot((data[0])[StartIndex:EndIndex])
         #plt.show()
         #Numerical Integration using Simpsons Rule
         Area = simps((data[0])[StartIndex:EndIndex],TimeArray[StartIndex:EndIndex])
         print("Intergrated Pulse: ",Area)
         print("Min Voltage = ",np.min(data[0]))
         PN = int(Area/PhotonCalibration)
         print("Number of photons = ", PN)
         if(i<1000)==0:
            TriggerDepthA.append(np.min(data[i]))
            PhotonNumberA.append(PN)
         else:
            TriggerDepthB.append(np.min(data[i]))
            PhotonNumberB.append(PN) 
    
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
