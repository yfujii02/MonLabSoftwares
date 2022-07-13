# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:21:33 2020
Plot Peak Values from overnight data collection and temperature

@author: smdek2
"""

import MonLabAnalysisFunctions as MLA
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import monashspa.PHS2061 as spa
import math
import os

#DataPath = r'C:\Users\smdek2\Documents\NovemberFibreTests\NovemberPlasticFibre\\'
DataPath = r'/home/comet/work/pico/data/'
Voltage = 42
MuFit=[]
Date = ['Nov26']
DataType = 0 #0: Peak Voltage, 1: Integrated Charge #2: LED Driver 
NBinsPeak = 300
RangePeak = np.array([-10, 990])
SinglePhotonPeakData=[]
WL470=[]
Time=[]
Runs = 1
FullList = MLA.GetFileList(DataPath,'Nov26','plastic',1,470,Voltage,0)
#print(FullList)        
for r in range(Runs):
    Time.append(r+1)
    List = MLA.GetFileList(DataPath,'Nov25','plastic',1,470,Voltage,0)
    ICData=[]
    for i in range(len(List)):
       File = List[i]
       print(File)
       #Time.append(os.path.getmtime(List[i]))
       #print(File)
       Waveform=MLA.GetWaveformData(File)
       MLA.GetData(Waveform,ICData,1,DataType,SinglePhotonPeakData)
                    
       muFit,muFitE,muRaw=MLA.GetHistogram(ICData,DataType,NBinsPeak, RangePeak[0], RangePeak[1],[200,60],1,'plastic',470,0)
       #WL470.append(muFit)
       if(i%10==0):
           ICData=[]
           WL470.append(muFit)
#TempFileData = np.array(np.loadtxt('COMET TL_Nov26_27.txt')) 
fig = plt.figure()
Time=7*np.array(Time)
plt.plot(WL470,label='Mean Peak Voltage')
#plt.plot(TimeT,Temp,label='Temperature')
plt.show()
