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
#matplotlib.use('agg')
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
    pData = []
    hData=[]
    Times = []
    count = 0
    MeanPeakVals=[]
    uMeanPeakVals=[]
    Chis=[]
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    print(folder)
 
    for f in folder:
        bins=[]
        vals=[]
        pData = []
        hData=[]
        
        myFunc.SetPolarity(waveform["Polarity"])
        myFunc.SetTimeScale(waveform["TimeScale"])
        myFunc.SetPeakThreshold(waveform["PeakThreshold"])
        myFunc.SetBins(histogram["NumberOfBins"],histogram["LowerRange"],histogram["UpperRange"])
        myFunc.SetSignalWindow(np.array(analysisWindow["Start"]),np.array(analysisWindow["Stop"]),np.array(analysisWindow["Baseline"]))
        myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
        myFunc.EnableFFTFilter(filtering["FFTCutoffFrequency"])
        myFunc.EnableBaselineFilter()
        myFunc.EnableTriggerCut(1,0,195)
        #Files = os.listdir(f)
        
        #Search = f 
        #os.chdir(Search)
        #Files = filter(os.path.isfile, os.listdir(Search))
        #Files = [os.path.join(Search, fi) for fi in Files] # add path to each file
        #Files.sort(key=lambda x: os.path.getmtime(x))
        
        #for file in Files:
        #    print(file)
        #    stat = os.path.getmtime(file)
        #    Times.append(stat)

        #    print("File time = ", stat)
        
        nch,trR,pData,hData, nEv = myFunc.AnalyseFolder(f,False)
        trR = np.array(trR)
        #if(count!=0):
        #    TriggerRs = np.concatenate((TriggerRs,trR))
        #else: TriggerRs = trR
        #count = count + 1
        #print(TriggerRs)
        #myFunc.PlotWaveformsFromAFile(f+"/data1.npy") 
        #plt.savefig("Test1.png")
       
        if len(bins)==0:
           for i in range(int(len(hData)/2)):
                bins.append(hData[2*i])
                vals.append(hData[2*i+1])
        else:
            for i in range(int(len(hData)/2)):
                vals[i] = vals[i]+hData[2*i+1]

        #For Irradiation Tests with Main Counter (Ch A) and Trigger Counter (ChB)
        MainC = pData[0]
        TrigC = pData[1]
        #print(len(pData[0]))
        #print(len(pData[1]))
        plt.figure()
        #H, xedges, yedges = np.histogram2d(TrigC, MainC, bins=250)
        #plt.pcolormesh(xedges, yedges, H, cmap='rainbow')
        plt.hist2d(MainC,TrigC,bins = (100,40), norm='log')
        plt.colorbar()
        plt.ylabel("Trigger Peak (mV)")
        plt.xlabel("Main Counter Peak (mV)")
        plt.savefig(str(FileNaming)+"_"+str(count)+"_TrigVMain.png")
        #os.system("mv ./"+str(FileNaming)+"TrigVMain* /mnt/c/Users/yfuj0004/work/MarchPMT_Tests/")
        #plt.figure()
        #Times = np.array(Times)
        #plt.scatter(Times-Times[0],TriggerRs)
        #plt.xlabel("Time (s)")
        #plt.ylabel("Trigger Rate (Hz)")
        #plt.show()
        

        fig, axs = plt.subplots(1,nch)
        #axs = axs.ravel()
        for i in range(len(bins)):
            #ax = fig.add_subplot(1, 2, 1 + i)
            axs[i].bar(bins[i][:-1],vals[i],width=bins[i][1]-bins[i][0],color='blue')
            #axs[i].set_ylim(bottom=0.2)
            #plt.yscale('log')
            axs[i].set_xlabel("Peak Voltage (mV)")
            axs[i].set_ylabel("Count")
            axs[i].set_title("Channel "+str(i))

            if histogram["GaussianFit"]==True and i==0:
                xdata = bins[i][:-1]
                ydata = vals[i]
                #ydata = myFunc.MovAvFilter(ydata)
                p_guess = [np.max(ydata),xdata[np.argmax(ydata)],0.25*xdata[np.argmax(ydata)]]
                # inc = (histogram["UpperRange"]-histogram["LowerRange"])/histogram["NumberOfBins"]
                # thresh_val = int(250/inc)
                # print("Fitting")
                # print(inc)
                # print(thresh_val)
                # popt,pcov = curve_fit(Gaus,xdata=xdata[:thresh_val],ydata=ydata[:thresh_val],p0=p_guess,maxfev=5000)
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
                axs[i].plot(xdata,Gaus(xdata,*popt),color='k',linestyle='--')
                #axs[i].text(350,np.max(ydata)+i*20,str(r'$\mu$=%.2f $\pm$ %.2f' % (popt[1],perr[1])),fontsize=18)
                
                diff = (ydata-Gaus(xdata,*popt))**2
                sigma = np.sqrt(ydata)
                for i in range(len(sigma)):
                    if(sigma[i]<=1): sigma[i]=1
                
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
                        
            if histogram["LandauFit"]==True and i==0:
                #plt.yscale('log')
                ### Try Laudau fitting
                #xdata = bins[i][:-1][bins[i][:-1]]
                #ydata = vals[i][bins[i][:-1]]
                #ydata = ydata[xdata<360]
                #xdata = xdata[xdata<360]
                xdata = bins[i][:-1]
                ydata = vals[i]
                pini  = [1000,175,1]
                popt,pcov =  curve_fit(Moyal, xdata=xdata,ydata=ydata,p0=pini,maxfev=5000)
                print("Updated Gaussian Fit Parameters")
                print(popt)
                print("Stat Uncertainty from Fitting")
                perr = np.sqrt(np.diag(pcov))
                print(perr)
                
                ymax = np.max(Moyal(xdata,*popt))
                axs[i].plot(xdata,Moyal(xdata,*popt),c='r',label='Landau')
                axs[i].text(350,ymax+i*20,str(r'$\mu$=%.2f $\pm$ %.2f' % (popt[1],perr[1])),fontsize=18)
                #plt.savefig(str(FileNaming)+'Landau_'+str(i)+'.png')
                
                diff = (ydata-Moyal(xdata,*popt))**2
                sigma = np.sqrt(ydata)
                for i in range(len(sigma)):
                    if(sigma[i]<=1): sigma[i]=1

            
                test_statistic = np.sum(diff/sigma**2)
                NDF = len(ydata) - len(popt)
                print("chisquare/NDF = {0:.2f} / {1:d} = {2:.2f}".format(test_statistic, NDF, test_statistic / float(NDF)))
                #os.system("mv ./"+str(FileNaming)+"Landau* /mnt/c/Users/yfuj0004/work/")
        plt.savefig(str(FileNaming)+'Gaussian_'+str(count)+'.png')
        #plt.show()
        count = count+1
    print("Data")
    print(MeanPeakVals)
    print(uMeanPeakVals)
    print(Chis)
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
