# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:08:40 2020

@author: smdek2
"""


import MonLabAnalysisFunctions as MLA
import numpy as np
import matplotlib.pyplot as plt
import math

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

uMuFitGlass1=[]
uMuFitGlass9=[]
uMuFitPlastic1=[]
uMuFitPlastic9=[]

for idate in range(len(Date)):
    for imat in range(len(FibreType)):
        for ilen in range(len(FibreLength)):
            for iwl in range(len(LEDWavelength)):
                List = MLA.GetFileList(DataPath,Date[idate],FibreType[imat],FibreLength[ilen],LEDWavelength[iwl],Voltage)
                #print("Filelist = ",List)
                ICData = []
                if(len(List)==0): continue
                #Skipping duplicates rerecorded later on...
                if(Date[idate]=='Sep21' and LEDWavelength[iwl]==405): continue
                if(Date[idate]=='Sep21' and LEDWavelength[iwl]==385 and FibreType[imat]=='glass'): continue
                if(Date[idate]=='Sep22' and LEDWavelength[iwl]==626): continue
                for i in range(len(List)):
                    File = List[i]
                    Waveform=MLA.GetWaveformData(File)
                    MLA.GetData(Waveform,ICData,FibreLength[ilen])
            
                muFit,muFitE,muRaw=MLA.GetHistogram(ICData, 200, -10, 190,[200,60],FibreLength[ilen], FibreType[imat], LEDWavelength[iwl],1)
                if(FibreType[imat]=='glass'):
                    if(FibreLength[ilen]==1):
                        MuFitGlass1.append(muFit)
                        uMuFitGlass1.append(muFitE)
                    else:
                        MuFitGlass9.append(muFit)
                        uMuFitGlass9.append(muFitE)
                elif(FibreType[imat]=='plastic'):
                    if(FibreLength[ilen]==1):
                        MuFitPlastic1.append(muFit)
                        uMuFitPlastic1.append(muFitE)
                    else: 
                        MuFitPlastic9.append(muFit)
                        uMuFitPlastic9.append(muFitE)

CompleteDataGlass=0
CompleteDataPlastic=0

if(len(MuFitGlass1)==len(MuFitGlass9)):CompleteDataGlass=1
if(len(MuFitPlastic1)==len(MuFitPlastic9)):CompleteDataPlastic=1

if(CompleteDataGlass==1):
    RatioGlass = np.array(MuFitGlass1)/np.array(MuFitGlass9)
    uRatioGlass = np.sqrt((np.array(uMuFitGlass1)/np.array(MuFitGlass9))**2+(np.array(uMuFitGlass9)*np.array(MuFitGlass1)/np.array(MuFitGlass9)**2)**2)
    AttGlass = 10.*np.log10(RatioGlass)*(1./9.08)*1000
    uAttGlass = 10*(1./(9.08))*1e3*np.log(math.e)*uRatioGlass/RatioGlass
    TransGlass = (1.-(1./9.08)*(1.-1./RatioGlass))*100

if(CompleteDataPlastic==1):
    RatioPlastic = np.array(MuFitPlastic1)/np.array(MuFitPlastic9)
    uRatioPlastic = np.sqrt((np.array(uMuFitPlastic1)/np.array(MuFitPlastic9))**2+(np.array(uMuFitPlastic9)*np.array(MuFitPlastic1)/np.array(MuFitPlastic9)**2)**2)
    AttPlastic = 10.*np.log10(RatioPlastic)*(1./8.30)*1000
    uAttPlastic = 10*(1./(8.30))*1e3*np.log(math.e)*uRatioPlastic/RatioPlastic
    TransPlastic = (1.-(1./8.30)*(1.-1./RatioPlastic))*100

fig1 = plt.figure()
if(CompleteDataGlass==1):plt.errorbar(LEDWavelength,AttGlass,uAttGlass,color='r',label='glass')
if(CompleteDataPlastic==1):plt.errorbar(LEDWavelength,AttPlastic,uAttPlastic,color='b',label='plastic')
plt.title('Attenuation')
plt.ylabel('Attenuation')
plt.xlabel('LED Wavelength (nm)')
plt.legend()
plt.show()

fig2 = plt.figure()
if(CompleteDataGlass==1):plt.plot(LEDWavelength,TransGlass,color='r',label='glass')
if(CompleteDataPlastic==1):plt.plot(LEDWavelength,TransPlastic,color='b',label='plastic')
plt.title('Transmission')
plt.ylabel('Transmission (%)')
plt.xlabel('LED Wavelength (nm)')
plt.legend()
plt.show()

print("Done")

