# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:08:40 2020

@author: smdek2
"""


import MonLabAnalysisFunctions as MLA
import numpy as np
import matplotlib.pyplot as plt

DataPath = r'/home/comet/work/pico/data/'
Date = ['Sep21','Sep22','Sep24']
FibreType = ['plastic','glass']
FibreLength = [1,9]
LEDWavelength =[385,405,470,525,585,626]
Voltage = 42

MuFitGlass1=[]
MuFitGlass9=[]
MuFitPlastic1=[]
MuFitPlastic9=[]

for idate in range(len(Date)):
    for imat in range(len(FibreType)):
        for ilen in range(len(FibreLength)):
            for iwl in range(len(LEDWavelength)):
                List = MLA.GetFileList(DataPath,Date[idate],FibreType[imat],FibreLength[ilen],LEDWavelength[iwl],Voltage)
                #print("Filelist = ",List)
                ICData = []
                if(len(List)==0): continue
                for i in range(len(List)):
                    File = List[i]
                    Waveform=MLA.GetWaveformData(File)
                    MLA.GetData(Waveform,ICData,FibreLength[ilen])
            
                muFit,muFitE,muRaw=MLA.GetHistogram(ICData, 200, -10, 190,[200,60],FibreLength[ilen], FibreType[imat], LEDWavelength[iwl],1)
                if(FibreType[imat]=='glass'):
                    if(FibreLength[ilen]==1):MuFitGlass1.append(muFit)
                    else: MuFitGlass9.append(muFit)
                elif(FibreType[imat]=='plastic'):
                    if(FibreLength[ilen]==1):MuFitPlastic1.append(muFit)
                    else: MuFitPlastic9.append(muFit)

CompleteDataGlass=0
CompleteDataPlastic=0

if(len(MuFitGlass1)==len(MuFitGlass9)):CompleteDataGlass=1
if(len(MuFitPlastic1)==len(MuFitPlastic9)):CompleteDataPlastic=1

if(CompleteDataGlass==1):
    RatioGlass = np.array(MuFitGlass1)/np.array(MuFitGlass9)
    AttGlass = 10.*np.log10(RatioGlass)*(1./8.)*1000

if(CompleteDataPlastic==1):
    RatioPlastic = np.array(MuFitPlastic1)/np.array(MuFitPlastic9)
    AttPlastic = 10.*np.log10(RatioPlastic)*(1./8.)*1000

fig1 = plt.figure()
if(CompleteDataGlass==1):plt.plot(LEDWavelength,AttGlass,color='r',label='glass')
if(CompleteDataPlastic==1):plt.plot(LEDWavelength,AttPlastic,color='b',label='plastic')
plt.title('Attenuation')
plt.ylabel('Attenuation')
plt.xlabel('LED Wavelength')
plt.legend()
plt.show()
print("Done")

