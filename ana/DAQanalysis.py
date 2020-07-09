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
for j in range(NumberFiles):
    File = AnalysisFiles[j]
    data = np.array(np.load(File))
    print(data.shape)
    print(File)
    print(data)
    print(len(data[1]))
    #plt.plot(data[0])
    #plt.show()
	
    NumberTriggers = len(data)
    TriggerDepthA = []
    TriggerDepthB = []
    print(NumberTriggers)

    for i in range(NumberTriggers):
        if(i<1000)==0:
            TriggerDepthA.append(np.min(data[i]))
        else:
            TriggerDepthB.append(np.min(data[i])) 
    
    AverageTriggerDepthA = np.average(TriggerDepthA)
    AverageTriggerDepthB = np.average(TriggerDepthB)
   # print(AverageTriggerDepth)
    AllTriggerDepthsA.append(AverageTriggerDepthA)
    AllTriggerDepthsB.append(AverageTriggerDepthB)

AllFileAverageDepthA = np.average(AllTriggerDepthsA)
AllFileAverageDepthB = np.average(AllTriggerDepthsB)
print("Average minimum voltage in Channel A = %.2f mV" %(AllFileAverageDepthA))
print("Average minimum voltage in Channel B = %.2f mV" %(AllFileAverageDepthB))

