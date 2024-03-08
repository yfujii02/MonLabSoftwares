#######################################################
#
# This is a minimum example script to run the analysis
# calling function to make a charge distribution plot
#
#######################################################

import sys
import os
import numpy as np
import MPPCAnalysisFunctions as myFunc
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from fitFunctions import Moyal,Gaus
from loadSettings import load_analysis_conf
import scipy.stats as stats
from scipy.stats import chisquare

folder=[]
conffile="analysis_settings.yaml"

def main():
    nch = 0 ### Number of channel, should be constant?
    trR = 0 ### Trigger Rate
    bins=[]
    vals=[]
    print("HERE")
    AllBins = []
    AllVals = []
    Times = []
    count = 0
    MeanPeakVals=[]
    uMeanPeakVals=[]
    Chis=[]
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    print(folder)
 
    for f in folder:
        pbins=[]
        pvals=[]
        tbins=[]
        tvals=[]
        pData=[]
        tData=[]

        myFunc.SetPolarity(waveform["Polarity"])
        myFunc.SetTimeScale(waveform["TimeScale"])
        myFunc.SetPeakThreshold(waveform["PeakThreshold"])
        myFunc.SetBins(histogram["BinSize"],histogram["LowerRange"],histogram["UpperRange"])
        myFunc.SetSignalWindow(np.array(analysisWindow["Start"]),np.array(analysisWindow["Stop"]),np.array(analysisWindow["Baseline"]))
        #myFunc.EnableDiffFilter(filtering["DiffPoints"])
        myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
        #myFunc.EnableFFTFilter(filtering["UpperFFTCutoffFrequency"],filtering["LowerFFTCutoffFrequency"])
        myFunc.EnableBaselineFilter()
        #myFunc.EnableTriggerCut(1,0,195)
        
        nch,trR,pData,tData,nEv = myFunc.AnalyseFolder(f,False)
        trR = np.array(trR)
       
        #if len(pbins)==0:
        for i in range(int(len(pData)/2)):
            pbins.append(pData[2*i])
            pvals.append(pData[2*i+1])
        #else:
        #    for i in range(int(len(pData)/2)):
        #        pvals[i] = pvals[i]+pData[2*i+1]
        
        #if len(tbins)==0:
        for i in range(int(len(tData)/2)):
            tbins.append(tData[2*i])
            tvals.append(tData[2*i+1])
        #else:
        #    for i in range(int(len(tData)/2)):
        #        tvals[i] = tvals[i]+tData[2*i+1]
        
        AllBins.append(pbins)
        AllVals.append(pvals)
        fig, axes = plt.subplots(1,2)
    
        for i in range(1):
            axes[i].bar(pbins[i][:-1],pvals[i],width=pbins[i][1]-pbins[i][0],color='blue')
            axes[i].set_xlabel("Peak Voltage (mV)")
            axes[i].set_ylabel("Count")
            axes[i].set_title("Channel "+str(i)) 
            
            #axes[i].set_ylim([0,30])
            #use axes[0,i] for one subplot, no idea why?

            if histogram["GaussianFit"][i]==True:
                xdata = pbins[i][:-1]
                ydata = pvals[i]
                p_guess = [np.max(ydata),xdata[np.argmax(ydata)],3]#0.25*xdata[np.argmax(ydata)]]
                popt,pcov = curve_fit(Gaus,xdata=xdata,ydata=ydata,sigma = np.sqrt(ydata), p0=p_guess,maxfev=5000)
                perr = np.sqrt(np.diag(pcov))
                print("Gaussian Fit Parameters")
                print(popt)
                print("Stat uncertainties")
                print(perr)
                #### Re-fit with updated parameters
                xdata = xdata[xdata<popt[1]+2.5*popt[2]]
                xdata2 = xdata[xdata>popt[1]-2.5*popt[2]]
                startIdx=0
                if (len(xdata2)<len(xdata)):
                    startIdx = len(xdata)-len(xdata2)
                #print(len(xdata))
                ydata = ydata[startIdx:len(xdata)]
                xdata = xdata[startIdx:]
                popt,pcov = curve_fit(Gaus,xdata=xdata,ydata=ydata,sigma = np.sqrt(ydata), p0=p_guess,maxfev=5000)
                print("Updated Gaussian Fit Parameters")
                print(popt)
                print("Stat Uncertainty from Fitting")
                perr = np.sqrt(np.diag(pcov))
                print(perr)
                axes[i].plot(xdata,Gaus(xdata,*popt),color='k',linestyle='--')
                
                diff = (ydata-Gaus(xdata,*popt))**2
                sigma = np.sqrt(ydata)
                for l in range(len(sigma)):
                    if(sigma[l]<=1): sigma[l]=1
                
                test_statistic = np.sum(diff/sigma**2)
                NDF = len(ydata) - len(popt)
                print("chisquare/NDF = {0:.2f} / {1:d} = {2:.2f}".format(test_statistic, NDF, test_statistic / float(NDF)))
                MeanPeakVals.append(popt[1])
                uMeanPeakVals.append(perr[1])
                Chis.append(test_statistic/float(NDF))
                #print("Chi-squared")
                #print(len(ydata))
                #print(len(Gaus(xdata,*popt)))
                #print(ydata)
                #print(Gaus(xdata,*popt))
                #chisquare(ydata,Gaus(xdata,*popt),ddof=len(popt),axis=None)
                                
                print(" ")
                print("Max Count value = ", np.max(ydata))
                print("Peak Voltage = ", xdata[np.argmax(ydata)])

                #Y_Model = Gaus(xdata, *popt)
                #r = ydata - Y_Model
                #chisq = np.sum((r/sig)**2)
                #df = len(ydata) - 2
                #print("chisq =",chisq,"df =",df,"chisq/df = ",round(chisq/df,2))
            
                #os.system("mv ./"+str(FileNaming)+"Gaussian* /mnt/c/Users/yfuj0004/work/")
                        
            if histogram["LandauFit"][i]==True:
                #plt.yscale('log')
                ### Try Laudau fitting
                #xdata = bins[i][:-1][bins[i][:-1]]
                #ydata = vals[i][bins[i][:-1]]
                #ydata = ydata[xdata<360]
                #xdata = xdata[xdata<360]
                xdata = pbins[i][:-1]
                ydata = pvals[i]
                pini  = [1000,175,1]
                popt,pcov =  curve_fit(Moyal, xdata=xdata,ydata=ydata,p0=pini,maxfev=5000)
                print("Updated Landau Fit Parameters")
                print(popt)
                print("Stat Uncertainty from Fitting")
                perr = np.sqrt(np.diag(pcov))
                print(perr)
                
                ymax = np.max(Moyal(xdata,*popt))
                axes[i].plot(xdata,Moyal(xdata,*popt),c='r',label='Landau, $\mu$'+str(i)+'= %.2f $\pm$ %.2f mV' % (popt[1],perr[1]))
                axes[i].legend()
                diff = (ydata-Moyal(xdata,*popt))**2
                sigma = np.sqrt(ydata)
                for i in range(len(sigma)):
                    if(sigma[i]<=1): sigma[i]=1

            
                test_statistic = np.sum(diff/sigma**2)
                NDF = len(ydata) - len(popt)
                print("chisquare/NDF = {0:.2f} / {1:d} = {2:.2f}".format(test_statistic, NDF, test_statistic / float(NDF)))
                #os.system("mv ./"+str(FileNaming)+"Landau* /mnt/c/Users/yfuj0004/work/")
        fig = plt.gcf()
        fig.set_size_inches(19.20,10.80)
        fig.tight_layout()
        fig.savefig(str(FileNaming)+'Fit_'+str(count)+'.png',bbox_inches='tight',dpi=200)
        count = count+1
        
        #fig, axes = plt.subplots(1,nch)
    
        #for i in range(nch):
        #    axes[i].bar(tbins[i][:-1],tvals[i],width=tbins[i][1]-tbins[i][0],color='blue')
        #    axes[i].set_xlabel("Edge Time (a.u.)")
        #    axes[i].set_ylabel("Count")
        #    axes[i].set_title("Channel "+str(i)) 
    #plt.show()
    
    #fig, axes = plt.subplots(1,2)
    #for i in range(len(AllBins)): 
    #    for j in range(2):
    #        axes[j].bar(AllBins[i][j][:-1],AllVals[i][j],width=AllBins[i][j][1]-AllBins[i][j][0],alpha=0.5,label = 'Dataset '+str(i))
    #        axes[j].set_xlabel("Peak Voltage (mV)")
    #        axes[j].set_ylabel("Count")
    #        axes[j].set_title("Channel "+str(j)) 

    #print("Data")
    #print(MeanPeakVals)
    #print(uMeanPeakVals)
    #print(Chis)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    conffile=args[1]
    FileNaming = args[2]
    FilePath = args[3]
    for i in range(len(args)-4):
        folder.append(args[3]+args[i+4])
    print(len(folder))
    main()
