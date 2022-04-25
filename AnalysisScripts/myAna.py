#######################################################
#
# This is a minimum example script to run the analysis
# calling function to make a charge distribution plot
#
#######################################################

import sys
import numpy as np
import MPPCAnalysisFunctions as myFunc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from fitFunctions import Moyal,Gaus
from loadSettings import load_analysis_conf

folder=[]
conffile="analysis_settings.yaml"

def main():
    nch = 0 ### Number of channel, should be constant?
    trR = 0 ### Trigger Rate
    bins=[]
    vals=[]
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    for f in folder:
        myFunc.SetPolarity(waveform["Polarity"])
        myFunc.SetBins(histogram["NumberOfBins"],histogram["LowerRange"],histogram["UpperRange"])
        myFunc.SetSignalWindow(analysisWindow["Start"],analysisWindow["Stop"],analysisWindow["Baseline"])
        myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
        myFunc.EnableFFTFilter(filtering["FFTCutoffFrequency"])
        myFunc.EnableBaselineFilter()
        nch, trR, hData = myFunc.AnalyseFolder(f,False)
        #plt.show()
        myFunc.PlotWaveformsFromAFile(f+"/data2.npy") 
        print(len(hData))
        if len(bins)==0:
            for i in range(int(len(hData)/2)):
                bins.append(hData[2*i])
                vals.append(hData[2*i+1])
        else:
            for i in range(int(len(hData)/2)):
                vals[i] = vals[i]+hData[2*i+1]
        #plt.close('all')
    for i in range(len(bins)):
        plt.figure()
        #print(len(bins[i]),len(vals[i]))
        #print(bins[i])
        #print(vals[i])
        plt.bar(bins[i][:-1],vals[i],width=bins[i][1]-bins[i][0],color='blue')
        plt.ylim(bottom=0.2)
        plt.yscale('log')

        if i<nch and i!=1  and histogram["GaussianFit"]==True:
           xdata = bins[i][:-1]
           ydata = vals[i]
           p_guess = [60,1700,10]
           popt,pcov = curve_fit(Gaus,xdata=xdata,ydata=ydata,p0=p_guess,maxfev=5000)
           perr = np.sqrt(np.diag(pcov))
           print(popt)
           print(perr)
           plt.plot(xdata,Gaus(xdata,*popt),color='k',linestyle='--')
                       
        if i==nch and histogram["LandauFit"]==True:
            plt.yscale('log')
            ### Try Laudau fitting
            xdata = bins[i][:-1][bins[i][:-1]>50]
            ydata = vals[i][bins[i][:-1]>50]
            ydata = ydata[xdata<360]
            xdata = xdata[xdata<360]
            pini  = [1000,175,1]
            popt,pcov =  curve_fit(Moyal, xdata=xdata,ydata=ydata,p0=pini,maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            print(popt)
            print(perr)
            ymax = np.max(Moyal(xdata,*popt))
            plt.plot(xdata,Moyal(xdata,*popt),c='r',label='Landau')
            plt.text(popt[1],1.5*ymax,str(r'$\mu$=%.2f $\pm$ %.2f' % (popt[1],perr[1])),fontsize=18)
    plt.show()

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    conffile=args[1]
    for i in range(len(args)-2):
        folder.append(args[i+2])
    print(len(folder))
    main()
