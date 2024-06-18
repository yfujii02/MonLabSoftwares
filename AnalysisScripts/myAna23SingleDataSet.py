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
from fitFunctions import Moyal,Gaus
from loadSettings import load_analysis_conf
import scipy.stats as stats
from scipy.stats import chisquare
plt.rcParams.update({'font.size': 20})
folder=[]
conffile="analysis_settings.yaml"

def GausFit(xdat,ydat):
    p_guess = [np.max(ydat),xdat[np.argmax(ydat)],1] #0.25*xdat[np.argmax(ydat)]]
    popt,pcov = curve_fit(Gaus,xdata=xdat,ydata=ydat,sigma = np.sqrt(ydat), p0=p_guess,maxfev=5000)
    perr = np.sqrt(np.diag(pcov))
    print("Gaussian Fit Parameters")
    print(popt)
    print("Stat uncertainties")
    print(perr)
    #### Re-fit with updated parameters
    xdat = xdat[xdat<popt[1]+5*popt[2]]
    xdat2 = xdat[xdat>popt[1]-5*popt[2]]
    
    startIdx=0
    if (len(xdat2)<len(xdat)):
        startIdx = len(xdat)-len(xdat2)
    
    ydat = ydat[startIdx:len(xdat)]
    xdat = xdat[startIdx:]
    
    popt,pcov = curve_fit(Gaus,xdata=xdat,ydata=ydat,sigma = np.sqrt(ydat), p0=p_guess,maxfev=5000)
    print("Updated Gaussian Fit Parameters")
    print(popt)
    print("Stat Uncertainty from Fitting")
    perr = np.sqrt(np.diag(pcov))
    print(perr)
    #axes[i].plot(xdata,Gaus(xdata,*popt),color='k',linestyle='--')
    
    diff = (ydat-Gaus(xdat,*popt))**2
    sigma = np.sqrt(ydat)
    for i in range(len(sigma)):
        if(sigma[i]<=1): sigma[i]=1
    
    test_statistic = np.sum(diff/sigma**2)
    NDF = len(ydat) - len(popt)
    chi2 = test_statistic/float(NDF)
    print("chisquare/NDF = {0:.2f} / {1:d} = {2:.2f}".format(test_statistic, NDF, test_statistic / float(NDF)))
    return xdat, ydat, popt, perr, chi2

def main():
    nch = 0 ### Number of channel, should be constant?
    trR = 0 ### Trigger Rate
    bins=[]
    vals=[]
    print("HERE")
    AllBins = []
    AllVals = []
    Gains = []
    uGains = []
    uGains2=[]
    Times = []
    count = 0
    MeanPeakVals=[[],[]]
    uMeanPeakVals=[[],[]]
    Chis=[]
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    print(folder)

    Distance = np.array([5,20,27.5,35]) #Order as processed
    FitVals = np.array([0,0,0,0])
    uFitVals = np.array([0,0,0,0])
    j=0
    TotalRData = []
    TotalTC = []
    for f in folder:
        pbins=[]
        pvals=[]
        tbins=[]
        tvals=[]
        pData=[]
        tData=[]

        rData=[]
        TC = 0

        myFunc.SetPolarity(waveform["Polarity"])
        myFunc.SetTimeScale(waveform["TimeScale"])
        myFunc.SetPeakThreshold(waveform["PeakThreshold"])
        myFunc.SetMaxPeakThreshold(waveform["MaxPeakThreshold"])
        myFunc.SetOffset(waveform["Offset"])
        myFunc.SetFName(str(Distance[j]))
        myFunc.SetTimeThreshold(waveform["TimeThreshold"])
        myFunc.SetBins(histogram["BinSize"],histogram["LowerRange"],histogram["UpperRange"])
        myFunc.SetSignalWindow(np.array(analysisWindow["Start"]),np.array(analysisWindow["Stop"]),np.array(analysisWindow["Baseline"]))
        #myFunc.EnableDiffFilter(filtering["DiffPoints"])
        myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
        myFunc.EnableFFTFilter(filtering["UpperFFTCutoffFrequency"],filtering["LowerFFTCutoffFrequency"])
        #myFunc.EnableBaselineFilter()
        #myFunc.EnableTriggerCut(1,0,195)
        
        nch,trR,pData,tData,nEv,rData,TC = myFunc.AnalyseFolder(f,False)
        trR = np.array(trR)
        TotalTC.append(TC)
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
        
        AllBins.append(tbins)
        AllVals.append(tvals)
        fig, axes = plt.subplots(1,4)
        ChannelTitles = ["Counter A", "Counter B", "Off-Axis Trigger", "Centre Trigger"]
        for i in range(nch):
            axes[i].bar(pbins[i][:-1],pvals[i],width=pbins[i][1]-pbins[i][0],color='blue',alpha = 0.8)
            axes[i].set_xlabel("Peak Voltage (mV)")
            axes[i].set_ylabel("Count")
            axes[i].set_title(ChannelTitles[i]) 
            
            #axes[i].set_ylim([0,30])
            #use axes[0,i] for one subplot, no idea why?

            if histogram["GaussianFit"][i]==True:
                xdata = pbins[i][:-1]
                ydata = pvals[i]
                p_guess = [np.max(ydata),xdata[np.argmax(ydata)],1]
                popt,pcov = curve_fit(Gaus,xdata=xdata,ydata=ydata,sigma = np.sqrt(ydata), p0=p_guess,maxfev=5000)
                perr = np.sqrt(np.diag(pcov))
                print("Gaussian Fit Parameters")
                print(popt)
                print("Stat uncertainties")
                print(perr)
                #### Re-fit with updated parameters
                xdata = xdata[xdata<(popt[1]+2.5*popt[2])]
                xdata2 = xdata[xdata>(popt[1]-2.5*popt[2])]
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
                #print(xdata)
                print("Moyal")
                print(len(ydata))
                
                Centre = np.argmax(ydata)
                FitWinHWL = 5
                FitWinHWU = 20
               
                xdata = xdata[Centre-FitWinHWL:Centre+FitWinHWU]
                ydata = ydata[Centre-FitWinHWL:Centre+FitWinHWU]
                yerr = np.sqrt(ydata)

                print(Centre)
                print(len(ydata))
                #print(xdata)

                pini  = [100,80,1]
                popt,pcov =  curve_fit(Moyal, xdata=xdata,ydata=ydata,sigma = yerr,p0=pini,maxfev=5000)
                print("Updated Landau Fit Parameters")
                print(popt)
                print("Stat Uncertainty from Fitting")
                perr = np.sqrt(np.diag(pcov))
                print(perr)
                
                ymax = np.max(Moyal(xdata,*popt))
                if(i==1): axes[i].set_xlim([0,100])
                else: axes[i].set_xlim([0,150])
                axes[i].plot(xdata,Moyal(xdata,*popt),c='k',linestyle = '--',label='Landau, $\mu$'+str(i)+'= %.2f $\pm$ %.2f mV' % (popt[1],perr[1]))
                #if(i==0):
                    # Shrink current axis's height by 10% on the bottom
                #box = axes[i].get_position()
                #axes[i].set_position([box.x0, box.y0 + box.height * 0.3,
                        #box.width, box.height * 0.7])
                if(i==0):
                    # Put a legend below current axis
                    axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                                fancybox=True, shadow=True, ncol=1)
                
                diff = (ydata-Moyal(xdata,*popt))**2
                sigma = np.sqrt(ydata)
                for l in range(len(sigma)):
                    if(sigma[l]<=1): sigma[l]=1

            
                test_statistic = np.sum(diff/sigma**2)
                NDF = len(ydata) - len(popt)
                print("chisquare/NDF = {0:.2f} / {1:d} = {2:.2f}".format(test_statistic, NDF, test_statistic / float(NDF)))
                print(MeanPeakVals)
                print(i)
                MeanPeakVals[i].append(popt[1])
                uMeanPeakVals[i].append(perr[1])
                #os.system("mv ./"+str(FileNaming)+"Landau* /mnt/c/Users/yfuj0004/work/")
        

        #GainPlots
        #xdat,ydat,Pop,Per,chi2 = GausFit(pbins[0][:-1],pvals[0])
        #plt.plot(xdat,Gaus(xdat,*Pop),color='k',linestyle='--')
        #plt.bar(pbins[0][:-1],pvals[0],width=pbins[0][1]-pbins[0][0],color='red',alpha = 0.5, label = 'Peak: $\mu_{V}$ = %.2f $\pm$ %0.2f mV' % (Pop[1], Per[1]))
        #plt.savefig(FileNaming+"_"+str(count)+".png")
        #Gains.append(Pop[1])
        #uGains.append(Pop[2])
        #uGains2.append(Per[1])
        #count = count+1
        #j=j+1
        #print("HERE")
        #continue

        fig = plt.gcf()
        fig.set_size_inches(19.20,10.80)
        fig.tight_layout()
        fig.savefig(str(Distance[j])+'_LandauFit.png',bbox_inches='tight',dpi=200)
        fig.savefig(FileNaming+'.png',bbox_inches='tight',dpi=200)
        
        fig = plt.figure()
        xdat,ydat,Pop,Per,chi2 = GausFit(tbins[0][:-1],tvals[0])
        xfit = np.linspace(0,15,500)
        plt.plot(xfit,Gaus(xfit,*Pop),color='k',linestyle='--')
        plt.bar(tbins[0][:-1],tvals[0],width=tbins[0][1]-tbins[0][0],color='red',alpha = 0.5, label = 'Counter A: $\sigma_{t}$ = %.2f $\pm$ %0.2f ns' % (Pop[2], Per[2]))
        
        xdat,ydat,Pop,Per,chi2 = GausFit(tbins[1][:-1],tvals[1])
        plt.plot(xfit,Gaus(xfit,*Pop),color='k',linestyle='--')
        plt.bar(tbins[1][:-1],tvals[1],width=tbins[1][1]-tbins[1][0],color='blue',alpha = 0.5,label = "Counter B: $\sigma_{t}$ = %.2f $\pm$ %0.2f ns" % (Pop[2], Per[2]))
        
        plt.xlabel("$\Delta$ t [ns]")
        plt.ylabel("Counts")
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(19.20,10.80)
        fig.tight_layout()
        fig.savefig(str(Distance[j])+'_TimingAvB.png',bbox_inches='tight',dpi=200)
        
        count = count+1
        j=j+1
    
    stop=0
    if(stop==1): 
        print("Peaks, uPeaks")
        print(Gains)
        print(uGains)
        print(uGains2)

        

    else:    
        fig, axes = plt.subplots(2,2)

        xdat,ydat,Pop,Per,chi2 = GausFit(AllBins[0][0][:-1],AllVals[0][0])
        axes[0,0].plot(np.linspace(0,17,500),Gaus(np.linspace(0,17,500),*Pop),color='k',linestyle='--')
        axes[0,0].bar(AllBins[0][0][:-1],AllVals[0][0],width=AllBins[0][0][1]-AllBins[0][0][0],color='red',alpha=0.5,label = '5 cm: $\sigma_{t}$ = %.2f $\pm$ %0.2f ns \n          $\mu_{t}$ = %.2f $\pm$ %0.2f ns' % (Pop[2], Per[2], Pop[1], Per[1]))
        
        xdat,ydat,Pop,Per,chi2 = GausFit(AllBins[1][0][:-1],AllVals[1][0])
        axes[0,1].plot(np.linspace(0,17,500),Gaus(np.linspace(0,17,500),*Pop),color='k',linestyle='--')
        axes[0,1].bar(AllBins[1][0][:-1],AllVals[1][0],width=AllBins[1][0][1]-AllBins[1][0][0],color='green',alpha = 0.5,label = '20 cm: $\sigma_{t}$ = %.2f $\pm$ %0.2f ns \n            $\mu_{t}$ = %.2f $\pm$ %0.2f ns' % (Pop[2], Per[2], Pop[1], Per[1]))
        
        xdat,ydat,Pop,Per,chi2 = GausFit(AllBins[2][0][:-1],AllVals[2][0])
        axes[1,0].plot(np.linspace(0,17,500),Gaus(np.linspace(0,17,500),*Pop),color='k',linestyle='--')
        axes[1,0].bar(AllBins[2][0][:-1],AllVals[2][0],width=AllBins[2][0][1]-AllBins[2][0][0],color='orange',alpha = 0.5,label = '27.5 cm: $\sigma_{t}$ = %.2f $\pm$ %0.2f ns \n               $\mu_{t}$ = %.2f $\pm$ %0.2f ns' % (Pop[2], Per[2], Pop[1], Per[1]))
        

        xdat,ydat,Pop,Per,chi2 = GausFit(AllBins[3][0][:-1],AllVals[3][0])
        axes[1,1].plot(np.linspace(0,17,500),Gaus(np.linspace(0,17,500),*Pop),color='k',linestyle='--')
        axes[1,1].bar(AllBins[3][0][:-1],AllVals[3][0],width=AllBins[3][0][1]-AllBins[3][0][0],color='blue',alpha = 0.5,label = '35 cm: $\sigma_{t}$ = %.2f $\pm$ %0.2f ns \n            $\mu_{t}$ = %.2f $\pm$ %0.2f ns' % (Pop[2], Per[2], Pop[1], Per[1]))
        
        #plt.xlabel("$\Delta$ t [ns]")
        axes[0,0].set_ylabel("Counts")
        axes[1,0].set_ylabel("Counts")
        axes[1,0].set_xlabel("$\Delta$ t [ns]")
        axes[1,1].set_xlabel("$\Delta$ t [ns]")
        axes[0,0].set_xlim([-5,20])
        axes[1,0].set_xlim([-5,20])
        axes[0,1].set_xlim([-5,20])
        axes[1,1].set_xlim([-5,20])
        #axes[0,0].legend()
        axes[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=1, fancybox=True, shadow=True)
        axes[1,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=1, fancybox=True, shadow=True)
        axes[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=1, fancybox=True, shadow=True)
        axes[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=1, fancybox=True, shadow=True)
        #axes[1,0].legend()
        #axes[0,1].legend()
        #axes[1,1].legend()
        fig = plt.gcf()
        fig.set_size_inches(19.20,10.80)
        fig.tight_layout()
        fig.savefig('CounterA_Timing.png',bbox_inches='tight',dpi=200)
        
        fig, axes = plt.subplots(2,2)
        xdat,ydat,Pop,Per,chi2 = GausFit(AllBins[0][1][:-1],AllVals[0][1])
        axes[0,0].plot(np.linspace(0,17,500),Gaus(np.linspace(0,17,500),*Pop),color='k',linestyle='--')
        axes[0,0].bar(AllBins[0][1][:-1],AllVals[0][1],width=AllBins[0][1][1]-AllBins[0][1][0],color='red',alpha=0.5,label = '5 cm: $\sigma_{t}$ = %.2f $\pm$ %0.2f ns \n          $\mu_{t}$ = %.2f $\pm$ %0.2f ns' % (Pop[2], Per[2], Pop[1], Per[1]))
        

        xdat,ydat,Pop,Per,chi2 = GausFit(AllBins[1][1][:-1],AllVals[1][1])
        axes[0,1].plot(np.linspace(0,17,500),Gaus(np.linspace(0,17,500),*Pop),color='k',linestyle='--')
        axes[0,1].bar(AllBins[1][1][:-1],AllVals[1][1],width=AllBins[1][1][1]-AllBins[1][1][0],color='green',alpha=0.5,label = '20 cm: $\sigma_{t}$ = %.2f $\pm$ %0.2f ns \n            $\mu_{t}$ = %.2f $\pm$ %0.2f ns' % (Pop[2], Per[2], Pop[1], Per[1]))
        
        xdat,ydat,Pop,Per,chi2 = GausFit(AllBins[2][1][:-1],AllVals[2][1])
        axes[1,0].plot(np.linspace(0,17,500),Gaus(np.linspace(0,17,500),*Pop),color='k',linestyle='--')
        axes[1,0].bar(AllBins[2][1][:-1],AllVals[2][1],width=AllBins[2][1][1]-AllBins[2][1][0],color='orange',alpha=0.5,label = '27.5 cm: $\sigma_{t}$ = %.2f $\pm$ %0.2f ns \n               $\mu_{t}$ = %.2f $\pm$ %0.2f ns' % (Pop[2], Per[2], Pop[1], Per[1]))

        xdat,ydat,Pop,Per,chi2 = GausFit(AllBins[3][1][:-1],AllVals[3][1])
        axes[1,1].plot(np.linspace(0,17,500),Gaus(np.linspace(0,17,500),*Pop),color='k',linestyle='--')
        axes[1,1].bar(AllBins[3][1][:-1],AllVals[3][1],width=AllBins[3][1][1]-AllBins[3][1][0],color='blue',alpha = 0.5,label = '35 cm: $\sigma_{t}$ = %.2f $\pm$ %0.2f ns \n            $\mu_{t}$ = %.2f $\pm$ %0.2f ns' % (Pop[2], Per[2], Pop[1], Per[1]))
    
        axes[0,0].set_ylabel("Counts")
        axes[1,0].set_ylabel("Counts")
        axes[1,0].set_xlabel("$\Delta$ t [ns]")
        axes[1,1].set_xlabel("$\Delta$ t [ns]")
        axes[0,0].set_xlim([-5,20])
        axes[1,0].set_xlim([-5,20])
        axes[0,1].set_xlim([-5,20])
        axes[1,1].set_xlim([-5,20])
        #axes[0,0].legend()
        #axes[1,0].legend()
        #axes[0,1].legend()
        #axes[1,1].legend()
        axes[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=1, fancybox=True, shadow=True)
        axes[1,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=1, fancybox=True, shadow=True)
        axes[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=1, fancybox=True, shadow=True)
        axes[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=1, fancybox=True, shadow=True)
        fig = plt.gcf()
        fig.set_size_inches(19.20,10.80)
        fig.tight_layout()
        fig.savefig('CounterB_Timing.png',bbox_inches='tight',dpi=200)
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
        #plt.legend()
        #plt.show()

        fig = plt.figure()
        plt.errorbar(Distance,MeanPeakVals[0],yerr=uMeanPeakVals[0],color='r',fmt='*',linestyle='')
        #plt.errorbar(Distance,MeanPeakVals[1],yerr=uMeanPeakVals[1],color='b',fmt='*',linestyle='')
        plt.xlabel("Distance from fibres [cm]")
        plt.ylabel("Landau Fit Peak [mV]")
        #plt.ylim([40,60])
        fig.tight_layout()
        fig.savefig('LandauMean.png',bbox_inches='tight',dpi=200)
        plt.show()
        print(TotalRData)
        print("Total Coinc")
        print(TotalTC)
        print(np.sum(TotalTC))
        #plt.figure()
        #plt.hist(TotalRData,bins=int((np.max(TotalRData)-np.min(TotalRData))/10),range=[np.min(TotalRData),np.max(TotalRData)])  
        #plt.xlabel("$T_R$ (ns)")
        #plt.ylabel("Counts")
        #plt.show()


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
