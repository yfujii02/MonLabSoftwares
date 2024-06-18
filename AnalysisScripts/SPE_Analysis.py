#######################################################
#
# This is a minimum example script to run the analysis
# calling function to make a charge distribution plot
#
#######################################################

import sys
import os
import numpy as np
import MPPCAnalysisFunctions23 as myFunc
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from fitFunctions import Moyal, Gaus, NGaus
from loadSettings import load_analysis_conf
import scipy.stats as stats
from scipy.stats import chisquare

folder=[]
conffile="analysis_settings.yaml"

def main():
    nCh = 0 ### Number of channel, should be constant?
    TriggerRates = []
    nEv = 0

    pData = []
    tData = []

    pbins = []
    pvals = []
    tvals = []
    tbins = []
    
    AllBins = []
    AllVals = []
    Times = []
    count = 0
    MeanPeakVals=[]
    uMeanPeakVals=[]
    Chis=[]
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    print(DataPath)
    


    myFunc.SetPolarity(waveform["Polarity"])
    myFunc.SetTimeScale(waveform["TimeScale"])
    myFunc.SetPeakThreshold(waveform["PeakThreshold"])
    #myFunc.SetMaxPeakThreshold(waveform["MaxPeakThreshold"])
    #myFunc.SetOffset(waveform["Offset"])
    myFunc.SetBins(histogram["BinSize"],histogram["LowerRange"],histogram["UpperRange"])
    myFunc.SetSignalWindow(np.array(analysisWindow["Start"]),np.array(analysisWindow["Stop"]),np.array(analysisWindow["Baseline"]))
    myFunc.EnableDiffFilter(filtering["DiffPoints"])
    #myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
    #myFunc.EnableFFTFilter(filtering["UpperFFTCutoffFrequency"],filtering["LowerFFTCutoffFrequency"])
    #myFunc.EnableBaselineFilter()
    #myFunc.EnableTriggerCut(1,0,195)
        
    nCh,TriggerRates, pData, nEv = myFunc.AnalyseFolder(DataPath,False)
    
    for i in range(int(len(pData)/2)):
        pbins.append(pData[2*i])
        pvals.append(pData[2*i+1])

    #For now data channel is assumed to be 0
    plt.close()
    plt.figure()
 
    #peaks, _ = find_peaks(pvals[0], distance = 20, height = 50)
    plt.bar(pbins[0][:-1],pvals[0],width=pbins[0][1]-pbins[0][0],color='blue', label = "Peak voltage data")
    #plt.scatter(pbins[0][:-1][peaks], pvals[0][peaks],s=20,color='k', marker='x')
    
    plt.xlabel("Peak Voltage (mV)")
    plt.ylabel("Count")

    if histogram["GaussianFit"]==True:
        
        ScaleFactor = 1

        xdata = pbins[0][:-1]*ScaleFactor
        ydata = pvals[0]
        
        peaks, _ = find_peaks(ydata, distance = 20, height = 50)
        PV = xdata[peaks]
        PC = ydata[peaks]

        if(len(peaks)<2):
            print("Not enough initial guess peaks observed")
            sys.exit()
        N_Peaks = len(peaks)
        print("%s peaks detected" % (N_Peaks))

        Width_Guess = PV[1] - PV[0]

        Width_Guess_Index = peaks[1] - peaks[0]

        Guess_Params = [PV[0], PC[0], Width_Guess/2]

        for i in range(N_Peaks-1):
            Guess_Params += [PV[0]+(i+1)*Width_Guess, PC[1]/(2**i), Width_Guess/2]
            FitWindowCutoff = peaks[0]+(i+2)*Width_Guess_Index
        
        print("Restrict Fit Window")
        print(FitWindowCutoff) 
        print(len(xdata))
        FitWindowX = xdata[0:FitWindowCutoff]
        FitWindowY = ydata[0:FitWindowCutoff]

        popt, pcov = curve_fit(NGaus,FitWindowX,FitWindowY,p0=Guess_Params)

        Updated_Guess = [popt[0],popt[1],popt[2]]
        for i in range(N_Peaks-1):
            Updated_Guess+=[popt[(i+1)*3],popt[(i+1)*3+1],popt[(i+1)*3+2]]
       
        popt,pcov = curve_fit(NGaus,FitWindowX,FitWindowY,p0=Updated_Guess) 
        p_sigma = np.sqrt(np.diag(pcov))
        print("Fit Parameters")
        print("--------------")
        print("Mean  Amp  Sig")
        print("%.2f +- %.2f | %.2f +- %.2f | %.2f +- %.2f" % (popt[0]/ScaleFactor, np.sqrt(np.diag(pcov)[0]/ScaleFactor), popt[1], p_sigma[1], popt[2], p_sigma[2]))
        for i in range(N_Peaks-1):
            print("%.2f +- %.2f | %.2f +- %.2f | %.2f +- %.2f" % (popt[(i+1)*3]/ScaleFactor, np.sqrt(np.diag(pcov)[(i+1)*3]/ScaleFactor), popt[(i+1)*3+1], p_sigma[(i+1)*3+1], popt[(i+1)*3+2], p_sigma[(i+1)*3+2]))
        print("--------------")
        SPE = popt[6]/ScaleFactor - popt[3]/ScaleFactor
        uSPE = np.sqrt(np.diag(pcov)[6]/ScaleFactor+np.diag(pcov)[3]/ScaleFactor)
        print("Single photon size with current conditions = %.2f +- %.2f" % (SPE, uSPE))      
        #print(popt)
        #p_sigma = np.sqrt(np.diag(pcov))
        #print(p_sigma)
        Fit = NGaus(xdata,*popt)
        plt.plot(xdata/ScaleFactor,Fit,'k--',linewidth=1, label = "Gaussian Fits => SPE = %.2f +- %.2f mV" % (SPE, uSPE))
        plt.legend()

    plt.show()

if __name__ == "__main__":
    args=sys.argv
    Nargs = 4
    #### name of the folder where you have the data to be analysed
    if(len(args)!=Nargs):
        print("To run this script for analysing single photon size of a set of waveforms run:")
        print(" ")
        print("python3 SPE_Analysis.py [1] [2] [3]")
        print(" ")
        print("where:")
        print("[1]: Analysis settings yaml file name")
        print("[2]: SaveName for any plots to save")
        print("[3]: path to the directory folder containing the waveform files (will run through every file in directory)")
        sys.exit()
    else:
        conffile=args[1]
        SaveName = args[2]
        DataPath = args[3]

    main()
