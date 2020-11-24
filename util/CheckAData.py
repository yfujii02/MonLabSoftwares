# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:08:40 2020

@author: smdek2
"""


import MonLabAnalysisFunctions as MLA
import numpy as np
import matplotlib.pyplot as plt
import sys

args=sys.argv

DataPath = r'/home/comet/work/pico/data/'
Date = 'Sep'+str(args[4])
FibreType = args[1]
FibreLength = int(args[2])
LEDWavelength =int(args[3])
Voltage = 42

List = MLA.GetFileList(DataPath,Date,FibreType,FibreLength,LEDWavelength,Voltage)
#print("Filelist = ",List)
ICData = []
if(len(List)==0): exit()
for i in range(len(List)):
    File = List[i]
    Waveform=MLA.GetWaveformData(File)
    MLA.GetData(Waveform,ICData,FibreLength)

muFit,muFitE,muRaw=MLA.GetHistogram(ICData, 200, -10, 190,[200,60],FibreLength, FibreType, LEDWavelength,1)
plt.show()
print("Done")

