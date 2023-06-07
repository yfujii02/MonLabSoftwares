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

counters=[]
conffile="analysis_settings.yaml"

def main():
    nch = 0 ### Number of channel, should be constant?
    trR = 0 ### Trigger Rate
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    print(counters)
    Pos = ['5','10','15','20','22.5','25']
    PosBins = [[],[],[],[],[],[]]
    PosVals = [[],[],[],[],[],[]]
    PosPeaks = [[],[],[],[],[],[]]
    Colours = ['red','yellow','blue','green']
    Styles = ['solid','dashed','dotted','dashdot']
    Thresh  = ['50','50','50','40']
    if(histogram['GaussianFit']==True): FitName = "Gaussian"
    elif(histogram['LandauFit']==True): FitName = "Landau"
    else: FitName = "_"
    MeanPeakVals=[]
    uMeanPeakVals=[]

    for c in counters:
        
        bins=[]
        vals=[]
        #print("HERE")
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
        CounterFolder = FilePath+c
        print(CounterFolder)
        subfolders = [ folder.path for folder in os.scandir(CounterFolder) if folder.is_dir() ]
        for f in subfolders:
            print(f)
            if (len(os.listdir(f)) != 0 ):
                
                nch,trR,pData,hData,nEv = myFunc.AnalyseFolder(f,False)
                for i in range(int(len(hData)/2)):
                    bins.append(hData[2*i])
                    vals.append(hData[2*i+1])

                PosBins[Pos.index(os.path.basename(str(f)))].append(np.array(bins)) 
                PosVals[Pos.index(os.path.basename(str(f)))].append(np.array(vals)) 
                PosPeaks[Pos.index(os.path.basename(str(f)))].append(pData) 

    #print(PosBins)
    #print(PosVals)
    #print(PosPeaks)
    #For Irradiation Tests with Main Counter (Ch A) and Trigger Counter (ChB)
    #print("Fin")
    
    for i in range(len(PosPeaks)):
        if(len(PosPeaks[i])!=0):
            for j in range(len(PosPeaks[i])):
                
                MainC = PosPeaks[i][j][0]
                TrigC = PosPeaks[i][j][1]
                
                #print(len(pData[0]))
                #print(len(pData[1]))
                plt.figure()
                #H, xedges, yedges = np.histogram2d(TrigC, MainC, bins=250)
                #plt.pcolormesh(xedges, yedges, H, cmap='rainbow')
                plt.hist2d(MainC,TrigC,bins = (50,20), range = [[0,500],[0,200]],norm='log')
                plt.colorbar()
                plt.ylabel("Trigger Peak (mV)")
                plt.xlabel("Main Counter Peak (mV)")
                plt.savefig("/Users/samdekkers/LabSoftware/Data/Plots/"+str(counters[j])+"_"+Pos[i]+"_TrigVMain.png")

  
    for k in range(len(PosBins)):
        if(len(PosBins[k])!=0):
            fig, axs = plt.subplots(1,nch)
            fig.subplots_adjust(right=0.7)
            for j in range(len(PosPeaks[k])):
                if(len(PosBins[k][j])!=0):
                    for i in range(nch):
                        axs[i].bar(PosBins[k][j][i][:-1],PosVals[k][j][i],width=PosBins[k][j][i][1]-PosBins[k][j][i][0],color=Colours[j],alpha=0.3,label=str(counters[j]))
                        #axs[i].set_ylim(bottom=0.2)
                        #plt.yscale('log')
                        axs[i].set_xlabel("Peak Voltage (mV)")
                        axs[i].set_ylabel("Count")
                        axs[i].set_title("Pos "+str(Pos[k])+" cm, Ch "+str(i))
                        
                        #Current Data
                        print(" ")
                        print("Position = ", Pos[k])
                        print("Counter = ", counters[j])
                        print("Channel = ", i)
                        print(" ")

                        if histogram["GaussianFit"]==True and i==0:
                            xdata = PosBins[k][j][i][:-1]
                            ydata = PosVals[k][j][i]
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
                            MeanPeakVals.append(popt[1])
                            print("Stat Uncertainty from Fitting")
                            perr = np.sqrt(np.diag(pcov))
                            uMeanPeakVals.append(perr[1])
                            print(perr)
                            
                            #axs[i].text(350,np.max(ydata)+i*20,str(r'$\mu$=%.2f $\pm$ %.2f' % (popt[1],perr[1])),fontsize=18)
                            
                            diff = (ydata-Gaus(xdata,*popt))**2
                            sigma = np.sqrt(ydata)
                            for p in range(len(sigma)):
                                if(sigma[p]<=1): sigma[p]=1
                            
                            test_statistic = np.sum(diff/sigma**2)
                            NDF = len(ydata) - len(popt)
                            print("chisquare/NDF = {0:.2f} / {1:d} = {2:.2f}".format(test_statistic, NDF, test_statistic / float(NDF)))            
                            FitLabel = "A = {:.2f}$\pm${:.2f}\n $\mu$ = {:.2f}$\pm${:.2f}\n $\sigma$ = {:.2f}$\pm${:.2f}\n $\chi^2$/ndf = {:.2f}/{:d} = {:.2f}".format(popt[0], perr[0],popt[1], perr[1],popt[2], perr[2], test_statistic, NDF,test_statistic/float(NDF))
                            axs[i].plot(xdata,Gaus(xdata,*popt),color='k',linestyle=Styles[j],label=FitLabel)
                            axs[i].legend(bbox_to_anchor=(2.25, 1), loc=2, borderaxespad=0.)
                            print(" ")
                            print("Max Count value = ", np.max(ydata))
                            print("Peak Voltage = ", xdata[np.argmax(ydata)])
                            print(" ")
                            
                                    
                        if histogram["LandauFit"]==True and i==0:
                            #plt.yscale('log')
                            ### Try Laudau fitting
                            #xdata = bins[i][:-1][bins[i][:-1]]
                            #ydata = vals[i][bins[i][:-1]]
                            #ydata = ydata[xdata<360]
                            #xdata = xdata[xdata<360]
                            xdata = PosBins[k][j][i][:-1]
                            ydata = PosVals[k][j][i]
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
            
            plt.savefig("/Users/samdekkers/LabSoftware/Data/Plots/Pos"+str(Pos[k])+"_"+str(FitName)+".png",bbox_inches='tight')
            plt.show()
    print("MeanPeakVals")
    print(MeanPeakVals)
    print(uMeanPeakVals)
if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    conffile=args[1]
    FilePath = args[2]
    for i in range(len(args)-3):
        print(args[i+3])
        counters.append(args[i+3])
    print(len(counters))
    main()
