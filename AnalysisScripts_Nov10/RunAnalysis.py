# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:08:40 2020
#NOTE: At current version this code is optimised for peak rather than integrated charge
so be warned....
@author: smdek2
"""


import MonLabAnalysisFunctions as MLA
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import monashspa.PHS2061 as spa
import math

#DataPath = r'/home/comet/work/pico/data/'
#DataPath = r'C:/Users/samde/Desktop/COMET/PlasticFibreTests/October/Oct/'
# DataPath = r'C:\Users\smdek2\Documents\OctoberFibreTests\Oct\\'
# Date = ['Oct1','Oct2','Oct6','Oct8']
# FibreType = ['plastic','glass']
# FibreLength = [1,3,6]
# LEDWavelength =[385,405,470,525,626]
#DataPath = r'C:\Users\smdek2\Documents\NovemberFibreTests\6mPlastic626nm\\'
DataPath = r'/home/comet/work/pico/data/'
Date = ['Nov11']
Run = ['_','2_','3_','4_','5_','6_','7_','8_','9_','10_','11_']
FibreType = ['plastic']
FibreLength = [3]
LEDWavelength =[626]

# FibreType = ['plastic']
# FibreLength = [3]
# LEDWavelength =[525]


#Uncertainty using 405 repeat measurements
#DataPath = r'C:\Users\smdek2\Documents\OctoberFibreTests\All405\\'
# DataPath = r'C:/Users/samde/Desktop/COMET/PlasticFibreTests/October/All405/'
# Date = ['Oct6']
# FibreType = ['plastic','glass']
# FibreLength = [1,3,6]
# LEDWavelength =[405]



# Date = ['Oct6']
# FibreType = ['plastic']
# FibreLength = [3]
# LEDWavelength =[405]

Voltage = 42

MuFitGlass1=[]
MuFitGlass3=[]
MuFitGlass6=[]
MuFitPlastic1=[]
MuFitPlastic3=[]
MuFitPlastic6=[]
List=[]

WL385=[]
WL405=[]
WL470=[]
WL525=[]
WL585=[]
WL626=[]

uWL385=[]
uWL405=[]
uWL470=[]
uWL525=[]
uWL585=[]
uWL626=[]

WL385Len=[]
WL405Len=[]
WL470Len=[]
WL525Len=[]
WL585Len=[]
WL626Len=[]

WL385g=[]
WL405g=[]
WL470g=[]
WL525g=[]
WL585g=[]
WL626g=[]

uWL385g=[]
uWL405g=[]
uWL470g=[]
uWL525g=[]
uWL585g=[]
uWL626g=[]

WL385Leng=[]
WL405Leng=[]
WL470Leng=[]
WL525Leng=[]
WL585Leng=[]
WL626Leng=[]


DataType = 0 #0: Peak Voltage, 1: Integrated Charge #2: LED Driver
NBinsPeak = 300
RangePeak = np.array([-10, 990])

SinglePhotonPeakData=[]

LEDDriverBaseline = []
LEDDriverPeak = []

for imat in range(len(FibreType)):
    for iwl in range(len(LEDWavelength)):
        for ilen in range(len(FibreLength)):    
            for idate in range(len(Date)):
                for irun in range(len(Run)):
                    List = MLA.GetFileList(DataPath,Date[idate],FibreType[imat],FibreLength[ilen],LEDWavelength[iwl],Voltage,Run[irun])
                    ICData = []
                    if(len(List)==0): continue
                    
                    for i in range(len(List)):
                        File = List[i]
                        print(File)
                        Waveform=MLA.GetWaveformData(File)
                        MLA.GetData(Waveform,ICData,FibreLength[ilen],DataType,SinglePhotonPeakData)
                
                    if(DataType==1): muFit,muFitE,muRaw=MLA.GetHistogram(ICData,DataType,100, -10, 190,[200,60],FibreLength[ilen], FibreType[imat], LEDWavelength[iwl],0)
                    if(DataType==0): muFit,muFitE,muRaw=MLA.GetHistogram(ICData,DataType,NBinsPeak, RangePeak[0], RangePeak[1],[200,60],FibreLength[ilen], FibreType[imat], LEDWavelength[iwl],0)
                                    
                    # if(FibreType[imat]=='glass'):
                    #     if(FibreLength[ilen]==1):MuFitGlass1.append(muFit)
                    #     elif(FibreLength[ilen]==3): MuFitGlass3.append(muFit)
                    #     else: MuFitGlass6.append(muFit)
                    # elif(FibreType[imat]=='plastic'):
                    #     if(FibreLength[ilen]==1):MuFitPlastic1.append(muFit)
                    #     elif(FibreLength[ilen]==3):MuFitPlastic3.append(muFit)
                    #     else: MuFitPlastic6.append(muFit)
                        
                    if(FibreType[imat]=='plastic'):           
                        if(LEDWavelength[iwl]==385):
                            WL385.append(muFit)
                            uWL385.append(muFitE)
                            WL385Len.append(FibreLength[ilen])
                        elif(LEDWavelength[iwl]==405):
                            WL405.append(muFit)
                            uWL405.append(muFitE)
                            WL405Len.append(FibreLength[ilen])
                        elif(LEDWavelength[iwl]==470):
                            WL470.append(muFit)
                            uWL470.append(muFitE)
                            WL470Len.append(FibreLength[ilen])
                        elif(LEDWavelength[iwl]==525):
                            WL525.append(muFit)
                            uWL525.append(muFitE)
                            WL525Len.append(FibreLength[ilen])
                        elif(LEDWavelength[iwl]==585):
                            WL585.append(muFit)
                            uWL585.append(muFitE)
                            WL585Len.append(FibreLength[ilen])
                        elif(LEDWavelength[iwl]==626):
                            WL626.append(muFit)
                            uWL626.append(muFitE)
                            WL626Len.append(FibreLength[ilen])
                            
                    if(FibreType[imat]=='glass'):           
                        if(LEDWavelength[iwl]==385):
                            WL385g.append(muFit)
                            uWL385g.append(muFitE)
                            WL385Leng.append(FibreLength[ilen])
                        elif(LEDWavelength[iwl]==405):
                            WL405g.append(muFit)
                            uWL405g.append(muFitE)
                            WL405Leng.append(FibreLength[ilen])
                        elif(LEDWavelength[iwl]==470):
                            WL470g.append(muFit)
                            uWL470g.append(muFitE)
                            WL470Leng.append(FibreLength[ilen])
                        elif(LEDWavelength[iwl]==525):
                            WL525g.append(muFit)
                            uWL525g.append(muFitE)
                            WL525Leng.append(FibreLength[ilen])
                        elif(LEDWavelength[iwl]==585):
                            WL585g.append(muFit)
                            uWL585g.append(muFitE)
                            WL585Leng.append(FibreLength[ilen])
                        elif(LEDWavelength[iwl]==626):
                            WL626g.append(muFit)
                            uWL626g.append(muFitE)
                            WL626Leng.append(FibreLength[ilen])
                    

# print("Filelist = ",List)
# CompleteDataGlass=0
# CompleteDataPlastic=0

# if(len(MuFitGlass1)==len(MuFitGlass6)):CompleteDataGlass=1
# if(len(MuFitPlastic1)==len(MuFitPlastic6)):CompleteDataPlastic=1

# if(CompleteDataGlass==1):
#     RatioGlass = np.array(MuFitGlass1)/np.array(MuFitGlass6)
#     AttGlass = 10.*np.log10(RatioGlass)*(1./8.)*1000

# if(CompleteDataPlastic==1):
#     RatioPlastic = np.array(MuFitPlastic1)/np.array(MuFitPlastic6)
#     AttPlastic = 10.*np.log10(RatioPlastic)*(1./8.)*1000

# fig1 = plt.figure()
# if(CompleteDataGlass==1):plt.plot(LEDWavelength,AttGlass,color='r',label='glass')
# if(CompleteDataPlastic==1):plt.plot(LEDWavelength,AttPlastic,color='b',label='plastic')
# plt.title('Attenuation')
# plt.ylabel('Attenuation')
# plt.xlabel('LED Wavelength')
# plt.legend()
# plt.show()
# print("Done")
                        

#Systematic uncertainty as measured from 405 nm set of data
Syst_Uncertainty = 0.0785 #percentage value                        
Bin_Uncertainty = (RangePeak[1]-RangePeak[0])/NBinsPeak #mV for peak voltage data
print("Bin uncertainty = ",Bin_Uncertainty)

fig4=plt.figure()
alpha=0.5
colour='red'    
#SinglePhotonPeakData = np.sort(SinglePhotonPeakData)[67962:]
CurrentNSP,CurrentBinsSP,_=plt.hist(SinglePhotonPeakData,bins=30,density = True,range = (0,25),alpha=alpha,color=colour,label='Data')
plt.title("Single Photon Peak Finding") 
plt.xlim(0,20)
plt.xlabel("Peak Voltage (mV)")

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.stats

centers = (0.5*(CurrentBinsSP[1:]+CurrentBinsSP[:-1]))
pars, cov = curve_fit(lambda SinglePhotonPeakData, mu, sig : scipy.stats.norm.pdf(SinglePhotonPeakData, loc=mu, scale=sig), centers,CurrentNSP, p0=[2,5])
plt.plot(centers, scipy.stats.norm.pdf(centers,*pars), 'k--',linewidth = 2, label='fit')
Mu = pars[0]
Mu_Uncertainty = np.sqrt(cov[0,0])

print("Fit: %0.2f +- %0.2f" %(Mu,Mu_Uncertainty))
## Mean value of an array
Mean = (np.array(SinglePhotonPeakData)).mean()
print("Mean of data array = ",Mean)

#Fitting
# Bin_UncertaintyMu = 25/30
# nonlinear_model = spa.make_lmfit_model("(1/np.sqrt(2*math.pi)*A1*np.exp(-(x-B1)**2/2*(A1**2)))+(1/np.sqrt(2*math.pi)*A2*np.exp(-(x-B2)**2/2*(A2**2)))")
# #Initial guesses based on actual values - A0 values are roughly half of total A0 values
# nonlinear_params = nonlinear_model.make_params(A1=2.0,B1=5.0,A2=2.0,B2=12.5)
# fit_results = spa.model_fit(nonlinear_model,nonlinear_params,x=centers,y=CurrentNSP,u_y=Bin_UncertaintyMu)
# fit_results_output = spa.get_fit_parameters(fit_results)
# y_fit = fit_results.best_fit
# u_y_fit = fit_results.eval_uncertainty(sigma=2)

# plt.plot(centers,y_fit,marker="None",linestyle="--",color="black",label = "Fit")

# plt.legend()



GlassLength = np.array([0.997,2.999,7.079])
PlasticLength = np.array([1.005,3.000,6.300])


#Uncertainties without mu (normalised)
ErrorBar = np.ones(len(PlasticLength)) #ones of length data
# #u385 = np.array(WL385)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL385)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)
# u405 = np.array(WL405)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL405)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)
# u470 = np.array(WL470)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL470)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)
# u525 = np.array(WL525)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL525)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)
# #u585 = np.array(WL585)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL585)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)
# u626 = np.array(WL626)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL626)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)

# u385g = np.array(WL385g)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL385g)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)
# u405g = np.array(WL405g)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL405g)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)
# u470g = np.array(WL470g)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL470g)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)
# u525g = np.array(WL525g)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL525g)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)
# #u585g = np.array(WL585g)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL585g)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)
# u626g = np.array(WL626g)*Syst_Uncertainty/Mu+np.sqrt((Bin_Uncertainty*ErrorBar/Mu)**2+(np.array(WL626g)*(Mu_Uncertainty+Bin_Uncertainty)/Mu**2)**2)

# u385 = np.sqrt(((np.array(WL385)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL385)))/WL385[0])**2+(np.array(WL385)*(np.array(WL385)[0]*Syst_Uncertainty+(Bin_Uncertainty+np.array(uWL385)[0]))/np.array(WL385)**2)**2)
# u405 = np.sqrt(((np.array(WL405)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL405)))/WL405[0])**2+(np.array(WL405)*(np.array(WL405)[0]*Syst_Uncertainty+(Bin_Uncertainty+np.array(uWL405)[0]))/np.array(WL405)**2)**2)
# u470 = np.sqrt(((np.array(WL470)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL470)))/WL470[0])**2+(np.array(WL470)*(np.array(WL470)[0]*Syst_Uncertainty+(Bin_Uncertainty+np.array(uWL470)[0]))/np.array(WL470)**2)**2)
# u525 = np.sqrt(((np.array(WL525)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL525)))/WL525[0])**2+(np.array(WL525)*(np.array(WL525)[0]*Syst_Uncertainty+(Bin_Uncertainty+np.array(uWL525)[0]))/np.array(WL525)**2)**2)
# u626 = np.sqrt(((np.array(WL626)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL626)))/WL626[0])**2+(np.array(WL626)*(np.array(WL626)[0]*Syst_Uncertainty+(Bin_Uncertainty+np.array(uWL626)[0]))/np.array(WL626)**2)**2)

u385 = np.array(WL385)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL385))
u405 = np.array(WL405)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL405))
u470 = np.array(WL470)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL470))
u525 = np.array(WL525)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL525))
#u585 = np.array(WL585)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL585))
u626 = np.array(WL626)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL626))

# u385g = np.sqrt(((np.array(WL385g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL385g)))/WL385g[0])**2+(np.array(WL385g)*(np.array(WL385g)[0]*Syst_Uncertainty+(Bin_Uncertainty+np.array(uWL385g)[0]))/np.array(WL385g)**2)**2)
# u405g = np.sqrt(((np.array(WL405g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL405g)))/WL405g[0])**2+(np.array(WL405g)*(np.array(WL405g)[0]*Syst_Uncertainty+(Bin_Uncertainty+np.array(uWL405g)[0]))/np.array(WL405g)**2)**2)
# u470g = np.sqrt(((np.array(WL470g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL470g)))/WL470g[0])**2+(np.array(WL470g)*(np.array(WL470g)[0]*Syst_Uncertainty+(Bin_Uncertainty+np.array(uWL470g)[0]))/np.array(WL470g)**2)**2)
# u525g = np.sqrt(((np.array(WL525g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL525g)))/WL525g[0])**2+(np.array(WL525g)*(np.array(WL525g)[0]*Syst_Uncertainty+(Bin_Uncertainty+np.array(uWL525g)[0]))/np.array(WL525g)**2)**2)
# u626g = np.sqrt(((np.array(WL626g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL626g)))/WL626g[0])**2+(np.array(WL626g)*(np.array(WL626g)[0]*Syst_Uncertainty+(Bin_Uncertainty+np.array(uWL626g)[0]))/np.array(WL626g)**2)**2)


u385g = np.array(WL385g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL385g))
u405g = np.array(WL405g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL405g))
u470g = np.array(WL470g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL470g))
u525g = np.array(WL525g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL525g))
#u585g = np.array(WL585g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL585g))
u626g = np.array(WL626g)*Syst_Uncertainty+(Bin_Uncertainty*ErrorBar+np.array(uWL626g))


#Log of data
logWL385=np.log(np.array(WL385))
logWL405=np.log(np.array(WL405))
logWL470=np.log(np.array(WL470))
logWL525=np.log(np.array(WL525))
#logWL585=np.log(np.array(WL585))
logWL626=np.log(np.array(WL626))

ulog385 = u385/WL385
ulog405 = u405/WL405
ulog470 = u470/WL470
ulog525 = u525/WL525
#ulog585 = u585/WL585
ulog626 = u626/WL626

logWL385g=np.log(np.array(WL385g))
logWL405g=np.log(np.array(WL405g))
logWL470g=np.log(np.array(WL470g))
logWL525g=np.log(np.array(WL525g))
#logWL585g=np.log(np.array(WL585g))
logWL626g=np.log(np.array(WL626g))

ulog385g = u385g/WL385g
ulog405g = u405g/WL405g
ulog470g = u470g/WL470g
ulog525g = u525g/WL525g
#ulog585g = u585g/WL585g
ulog626g = u626g/WL626g

#Normalise
# WL385 = np.array(WL385)/np.array(WL385)[0]
# WL405 = np.array(WL405)/np.array(WL405)[0]
# WL470 = np.array(WL470)/np.array(WL470)[0]
# WL525 = np.array(WL525)/np.array(WL525)[0]
# WL626 = np.array(WL626)/np.array(WL626)[0]

# WL385g = np.array(WL385g)/np.array(WL385g)[0]
# WL405g = np.array(WL405g)/np.array(WL405g)[0]
# WL470g = np.array(WL470g)/np.array(WL470g)[0]
# WL525g = np.array(WL525g)/np.array(WL525g)[0]
# WL626g = np.array(WL626g)/np.array(WL626g)[0]

loglen=np.log(np.array(WL405Len))

Attenuation = []
AttError =[]
WavelengthPlot = np.array([385,405,470,525])
WavelengthPlotg = np.array([385,405,470,626])

def func(x, A, Lambda):
    return A*np.exp(-x/Lambda)

popt385, pcov385 = curve_fit(func, PlasticLength, WL385,sigma = u385,bounds=(0, [100000, 100000]))    
print("385 nm Attenuation Fit: ",popt385[1])
perr385 = np.sqrt(np.diag(pcov385))
Attenuation.append(popt385[1])
AttError.append(perr385[1])

popt405, pcov405 = curve_fit(func, PlasticLength, WL405,sigma = u405,bounds=(0, [100000, 100000]))    
print("405 nm Attenuation Fit: ",popt405[1])
perr405 = np.sqrt(np.diag(pcov405))
Attenuation.append(popt405[1])
AttError.append(perr405[1])

popt470, pcov470 = curve_fit(func,PlasticLength, WL470,sigma = u470,bounds=(0, [100000, 100000]))    
print("470 nm Attenuation Fit: ",popt470[1])
perr470 = np.sqrt(np.diag(pcov470))
Attenuation.append(popt470[1])
AttError.append(perr470[1])

popt525, pcov525 = curve_fit(func, PlasticLength, WL525,sigma = u525,bounds=(0, [100000, 100000]))    
print("525 nm Attenuation Fit: ",popt525[1])
perr525 = np.sqrt(np.diag(pcov525))
Attenuation.append(popt525[1])
AttError.append(perr525[1])

popt626, pcov626 = curve_fit(func, PlasticLength, WL626,sigma = u626,bounds=(0, [100000, 100000]))    
print("626 nm Attenuation Fit: ",popt626[1])
perr626 = np.sqrt(np.diag(pcov626))
#Attenuation.append(popt626[1])
#AttError.append(perr626[1])

fig2 = plt.figure()
plt.errorbar(PlasticLength,np.array(WL385),u385,ls='None',color='purple',label='385 nm')
plt.plot(PlasticLength,popt385[0]*np.exp(-(1.0/popt385[1])*PlasticLength),ls='--',color='purple',label='Fit 385 nm')
plt.errorbar(PlasticLength,np.array(WL405),u405,ls='None',color='blue',label='405 nm')
plt.plot(PlasticLength,popt405[0]*np.exp(-(1.0/popt405[1])*PlasticLength),ls='--',color='blue',label='Fit 405 nm')
plt.errorbar(PlasticLength,np.array(WL470),u470,ls='None',color='aqua',label='470 nm')
plt.plot(PlasticLength,popt470[0]*np.exp(-(1.0/popt470[1])*PlasticLength),ls='--',color='aqua',label='Fit 470 nm')
plt.errorbar(PlasticLength,np.array(WL525),u525,ls='None',color='green',label='525 nm')
plt.plot(PlasticLength,popt525[0]*np.exp(-(1.0/popt525[1])*PlasticLength),ls='--',color='green',label='Fit 525 nm')
#plt.errorbar(WL585Len,np.array(WL585),u585,color='yellow',label='585 nm')
plt.errorbar(PlasticLength,np.array(WL626),u626,ls='None',color='red',label='626 nm')
plt.plot(PlasticLength,popt626[0]*np.exp(-(1.0/popt626[1])*PlasticLength),ls='--',color='red',label='Fit 626 nm')

if(DataType==0):
    plt.title("Peak Voltage vs Fibre Length (Plastic)")
    plt.xlabel("Fibre Length (m)")
    plt.ylabel("Peak Voltage (mV))")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
if(DataType==1):
    plt.title("IC vs Fibre Length (Plastic)")
    plt.xlabel("Fibre Length (m)")
    plt.ylabel("IC")
    plt.legend()

Attenuationg = []
AttErrorg =[]

popt385g, pcov385g = curve_fit(func, GlassLength, WL385g,sigma = u385g,bounds=(0, [100000, 100000]))    
print("385 nm Attenuation Fit: ",popt385g[1])
perr385g = np.sqrt(np.diag(pcov385g))
Attenuationg.append(popt385g[1])
AttErrorg.append(perr385g[1])

popt405g, pcov405g = curve_fit(func, GlassLength, WL405g,sigma = u405g,bounds=(0, [100000, 100000]))    
print("405 nm Attenuation Fit: ",popt405g[1])
perr405g = np.sqrt(np.diag(pcov405g))
Attenuationg.append(popt405g[1])
AttErrorg.append(perr405g[1])

popt470g, pcov470g = curve_fit(func,GlassLength, WL470g,sigma = u470g,bounds=(0, [100000, 100000]))    
print("470 nm Attenuation Fit: ",popt470g[1])
perr470g = np.sqrt(np.diag(pcov470g))
Attenuationg.append(popt470g[1])
AttErrorg.append(perr470g[1])

popt525g, pcov525g = curve_fit(func, GlassLength, WL525g,sigma = u525g,bounds=(0, [100000, 100000]))    
print("525 nm Attenuation Fit: ",popt525g[1])
perr525g = np.sqrt(np.diag(pcov525g))
Attenuationg.append(popt525g[1])
AttErrorg.append(perr525g[1])

popt626g, pcov626g = curve_fit(func, GlassLength, WL626g,sigma = u626g,bounds=(0, [100000, 100000]))    
print("626 nm Attenuation Fit: ",popt626g[1])
perr626g = np.sqrt(np.diag(pcov626g))
Attenuationg.append(popt626g[1])
AttErrorg.append(perr626g[1])

fig3 = plt.figure()
plt.errorbar(GlassLength,np.array(WL385g),u385g,ls='None',color='purple',label='385 nm')
plt.plot(GlassLength,popt385g[0]*np.exp(-(1.0/popt385g[1])*GlassLength),ls='--',color='purple',label='Fit 385 nm')
plt.errorbar(GlassLength,np.array(WL405g),u405g,ls='None',color='blue',label='405 nm')
plt.plot(GlassLength,popt405g[0]*np.exp(-(1.0/popt405g[1])*GlassLength),ls='--',color='blue',label='Fit 405 nm')
plt.errorbar(GlassLength,np.array(WL470g),u470g,ls='None',color='aqua',label='470 nm')
plt.plot(GlassLength,popt470g[0]*np.exp(-(1.0/popt470g[1])*GlassLength),ls='--',color='aqua',label='Fit 470 nm')
plt.errorbar(GlassLength,np.array(WL525g),u525g,ls='None',color='green',label='525 nm')
plt.plot(GlassLength,popt525g[0]*np.exp(-(1.0/popt525g[1])*GlassLength),ls='--',color='green',label='Fit 525 nm')
#plt.errorbar(WL585Leng,np.array(WL585g),u585g,color='yellow',label='585 nm')
plt.errorbar(GlassLength,np.array(WL626g),u626g,ls='None',color='red',label='626 nm')
plt.plot(GlassLength,popt626g[0]*np.exp(-(1.0/popt626g[1])*GlassLength),ls='--',color='red',label='Fit 626 nm')

if(DataType==0):
    plt.title("Peak Voltage vs Fibre Length (Glass)")
    plt.xlabel("Fibre Length (m)")
    plt.ylabel("Peak Voltage (mV)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
if(DataType==1):
    plt.title("IC vs Fibre Length (Glass)")
    plt.xlabel("Fibre Length (m)")
    plt.ylabel("IC")
    plt.legend()
   
figatt = plt.figure()
plt.plot(WavelengthPlot,np.array(Attenuation),marker='x',ls='None',color='black',label='Plastic Fibre')
plt.errorbar(WavelengthPlot,np.array(Attenuation),np.array(AttError),ls='None',color='black')
#plt.ylim(0,160)
#plt.plot(WavelengthPlotg,np.array(Attenuationg),marker='x',ls='None',color='red',label='Glass Fibre')
#plt.errorbar(WavelengthPlotg,np.array(Attenuationg),np.array(AttErrorg),ls='None',color='red')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Attenuation Length (m)')
plt.title('Fibre Attenuation Lengths')
plt.legend()    

# #####################################################################################
# #Log Plots
# ErrorBar = np.log(Mu*np.ones(len(WL405Len)))/Mu*np.ones(len(WL405Len))

# fit385 = np.polyfit(loglen,logWL385,1)
# fit405 = np.polyfit(loglen,logWL405,1)
# fit470 = np.polyfit(loglen,logWL470,1)
# fit525 = np.polyfit(loglen,logWL525,1)
# #fit585 = np.polyfit(loglen,logWL585,1)
# fit626 = np.polyfit(loglen,logWL626,1)

# fit385g = np.polyfit(loglen,logWL385g,1)
# fit405g = np.polyfit(loglen,logWL405g,1)
# fit470g = np.polyfit(loglen,logWL470g,1)
# fit525g = np.polyfit(loglen,logWL525g,1)
# #fit585g = np.polyfit(loglen,logWL585g,1)
# fit626g = np.polyfit(loglen,logWL626g,1)
# print("###########################")
# print("Plastic")
# print("385 nm curve fit: ",fit385)
# print("405 nm curve fit: ",fit405)
# print("470 nm curve fit: ",fit470)
# print("525 nm curve fit: ",fit525)
# #print("585 nm curve fit: ",fit585)
# print("626 nm curve fit: ",fit626)
# print("Glass")
# print("385 nm curve fit: ",fit385g)
# print("405 nm curve fit: ",fit405g)
# print("470 nm curve fit: ",fit470g)
# print("525 nm curve fit: ",fit525g)
# #print("585 nm curve fit: ",fit585g)
# print("626 nm curve fit: ",fit626g)
# print("###########################")

# fig5 = plt.figure()
# plt.errorbar(loglen,logWL385,ulog385,color='purple',label='385 nm')
# plt.errorbar(loglen,logWL405,ulog405,color='blue',label='405 nm')
# plt.errorbar(loglen,logWL470,ulog470,color='aqua',label='470 nm')
# plt.errorbar(loglen,logWL525,ulog525,color='green',label='525 nm')
# #plt.errorbar(loglen,logWL585,ulog585,color='yellow',label='585 nm')
# plt.errorbar(loglen,logWL626,ulog626,color='red',label='626 nm')

# if(DataType==0):
#     plt.title("Peak Voltage vs Fibre Length (Plastic)")
#     plt.xlabel("Log Fibre Length")
#     plt.ylabel("Log Peak Voltage")
#     plt.legend()
    
# if(DataType==1):
#     plt.title("IC vs Fibre Length (Plastic)")
#     plt.xlabel("Log Fibre Length")
#     plt.ylabel("Log IC")
#     plt.legend()

# ErrorBar =np.log(Mu*np.ones(len(WL405Leng)))/Mu*np.ones(len(WL405Leng))

# fig6 = plt.figure()
# plt.errorbar(loglen,logWL385g,ulog385g,color='purple',label='385 nm')
# plt.errorbar(loglen,logWL405g,ulog405g,color='blue',label='405 nm')
# plt.errorbar(loglen,logWL470g,ulog470g,color='aqua',label='470 nm')
# plt.errorbar(loglen,logWL525g,ulog525g,color='green',label='525 nm')
# #plt.errorbar(loglen,logWL585g,ulog585g,color='yellow',label='585 nm')
# plt.errorbar(loglen,logWL626g,ulog626g,color='red',label='626 nm')

# if(DataType==0):
#     plt.title("Peak Voltage vs Fibre Length (Glass)")
#     plt.xlabel("Log Fibre Length")
#     plt.ylabel("Log Peak Voltage")
#     plt.legend()
    
# if(DataType==1):
#     plt.title("IC vs Fibre Length (Glass)")
#     plt.xlabel("Log Fibre Length")
#     plt.ylabel("Log IC")
#     plt.legend()

