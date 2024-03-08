#######################################################
#
# This is a minimum example script to run the analysis
# calling function to make a charge distribution plot
#
#######################################################

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from iminuit import Minuit
from probfit import UnbinnedLH, gaussian


### user defined functions
import MPPCAnalysisFunctions as myFunc
from fitFunctions import Moyal,Gaus,gauss_pdf,pois_gauss_pdf
from loadSettings import load_analysis_conf

folder=[]
conffile="analysis_settings.yaml"
#debug = True
debug = False

# Create an unbinned likelihood object

def fit_twice_wG(data):
    frange = [data.mean()-1.8*data.std(),data.mean()+1.8*data.std()]
    dataF = data[data>frange[0]]
    dataF = dataF[dataF<frange[0]]
    unbinned_likelihood = UnbinnedLH(gauss_pdf, dataF)
    # Initialise Minuit with initial parameter guesses and the likelihood function
    minuit = Minuit(unbinned_likelihood, mu=data.mean(), sigma=data.std())
    # Perform the fit
    minuit.migrad()
    frange = [minuit.values["mu"]-1.8*minuit.values["sigma"],minuit.values["mu"]+1.8*minuit.values["sigma"]]
    dataF = data[data>frange[0]]
    dataF = dataF[dataF<frange[1]]
    # 2nd attempt
    unbinned_likelihood2 = UnbinnedLH(gauss_pdf, dataF)
    minuit2 = Minuit(unbinned_likelihood, mu=minuit.values["mu"], sigma=minuit.values["sigma"])
    minuit2.migrad()
    for par in minuit2.values:
        print(par,":",minuit2.values[par],"+-",minuit2.errors[par])
    return minuit2

def fit_pgauss(data,gain,sigma0,sigma1):
    unbinned_likelihood = UnbinnedLH(pois_gauss_pdf, data)
    # Initialise Minuit with initial parameter guesses and the likelihood function
    minuit = Minuit(unbinned_likelihood, mu=0, plambda=1.5, gain=gain, sigma0=sigma0, sigma1=sigma1, nlim=10,
                    limit_mu=(-0.3*gain,0.3*gain),limit_plambda=(0,10),limit_gain=(0.5*gain,2*gain),limit_sigma0=(0.1*sigma0,10*sigma0),limit_sigma1=(0.1*sigma1,10*sigma1))
    minuit.fixed["plambda"]=True
    minuit.fixed["mu"]=True
    minuit.fixed["gain"]=True
    minuit.fixed["sigma0"]=True
    minuit.fixed["sigma1"]=True
    minuit.fixed["nlim"]=True
    #minuit.limits["plambda"]=(0, 50)
    #minuit.limits["gain"]=(0, None)
    #minuit.limits["sigma0"]=(0, None)
    #minuit.limits["sigma1"]=(0, None)
    # Perform the fit
    minuit.migrad()
    for par in minuit.values:
        print(par,":",minuit.values[par],"+-",minuit.errors[par])
    return minuit

def main():
    nch = 0 ### Number of channel, should be constant?
    bins=[]
    vals=[]
    print("HERE")
    AllBins = []
    AllVals = []
    Times = []
    MeanPeakVals=[]
    uMeanPeakVals=[]
    Chis=[]
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    print(folder)

    nfiles = 0
    hMean = [[],[],[]]
    hError= [[],[],[]]
    cMean = [[],[],[]]
    cError= [[],[],[]]
    for f in folder:

        myFunc.SetPolarity(waveform["Polarity"])
        myFunc.EnableChannels(waveform["ReadChannels"])
        myFunc.SetTimeScale(waveform["TimeScale"])
        myFunc.SetOffset(waveform["ChannelOffset"])
        #myFunc.SetRemoveNoisyEvents(0.78)
        myFunc.SetPeakThreshold(waveform["PeakThreshold"])
        myFunc.SetBins(histogram["BinSize"],histogram["LowerRange"],histogram["UpperRange"])
        myFunc.SetSignalWindow(np.array(analysisWindow["Start"]),np.array(analysisWindow["Stop"]),np.array(analysisWindow["Baseline"]))
        myFunc.SetIntegrationWindows(np.array(analysisWindow["IntegrationStart"]),np.array(analysisWindow["IntegrationEnd"]))
        myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
        #myFunc.EnableFFTFilter(filtering["UpperFFTCutoffFrequency"],filtering["LowerFFTCutoffFrequency"])
        #myFunc.EnableBaselineFilter()
       
        # chData contains all basic pulse information such as pulse height, charge and timing
        if debug:
            chData,nch,readCh,nEv = myFunc.AnalyseFolder(f,False,0,5)
        else:
            chData,nch,readCh,nEv = myFunc.AnalyseFolder(f,False)

        title=[["Peak Voltage [mV]","Charge [mV*ns]"],
               ["Charge [mV*ns]","Count"],
               ["Peak Voltage [mV]","Count"],
               ["Edge Time [ns]","Count"]]
        ctitle = ["Charge vs Height","Charge","Height","Time"]

        #### for 1 p.e. analysis
        #hAxis = np.linspace( -3, 21,144)
        #cAxis = np.linspace(-15,285,200)
        #tAxis = np.linspace(400,525,250)
        #### for linearity check
        hAxis = np.linspace(-10,100,440)
        cAxis = np.linspace(-15,1485,1000)
        tAxis = np.linspace(400,525,250)
        #for ch in range(2): ### for 1 p.e. analysis
        nfiles = nfiles + 1
        for ch in range(3):
            if (waveform["ReadChannels"][ch]==False): continue
            if ch==2:
                hAxis = np.linspace( -5,  85,360)
                cAxis = np.linspace(-15,3985,1500)
                tAxis = np.linspace(400, 800,400)
            fig, axes = plt.subplots(2,2)
            heightArray = myFunc.GetHeightArray(chData[0][ch])
            chargeArray = myFunc.GetChargeArray(chData[0][ch])
            timeArray   = myFunc.GetTimeArray(chData[0][ch])
            heightArray = np.array(heightArray)
            chargeArray = np.array(chargeArray)
            timeArray   = np.array(  timeArray)

            #hRes = fit_twice_wG(heightArray)
            #cRes = fit_twice_wG(chargeArray)
            if ch==2:
                hRes = fit_pgauss(heightArray,0.5,0.2,0.15)
                cRes = fit_pgauss(chargeArray,10,30,25)
            else:
                hRes = fit_pgauss(heightArray,2.4,0.3,0.2)
                cRes = fit_pgauss(chargeArray,28,8,5)

            tCut = [200,900]
            if ch<2:
                tCut = [416,436]

            hAxis2 = np.linspace(hAxis[0],hAxis[-1],10*len(hAxis))
            cAxis2 = np.linspace(cAxis[0],cAxis[-1],10*len(cAxis))
            tAxis2 = np.linspace(tAxis[0],tAxis[-1],10*len(tAxis))
            for i in range(4):
                j=int(i/2)
                if i==0:
                    axes[j,i%2].hist2d(heightArray[timeArray<tCut[1]],chargeArray[timeArray<tCut[1]],bins=[hAxis,cAxis],cmap=plt.cm.jet,cmin=0.5)
                if i==1:
                    axes[j,i%2].hist(chargeArray[timeArray<tCut[1]],bins=cAxis,density=True,alpha=0.8)
                    #axes[j,i%2].plot(cAxis2, gauss_pdf(cAxis2, *cRes.values[:]), label='mu='+"{:6.2f}".format(cRes.values["mu"]), linewidth=2)
                    axes[j,i%2].plot(cAxis2, pois_gauss_pdf(cAxis2, *cRes.values[:]), label='plambda='+"{:6.2f}".format(cRes.values["plambda"]), linewidth=2)
                if i==2:
                    axes[j,i%2].hist(heightArray[timeArray<tCut[1]],bins=hAxis,density=True,alpha=0.8)
                    #axes[j,i%2].plot(hAxis2, gauss_pdf(hAxis2, *hRes.values[:]), label='mu='+"{:6.2f}".format(hRes.values["mu"]), linewidth=2)
                    axes[j,i%2].plot(hAxis2, pois_gauss_pdf(hAxis2, *hRes.values[:]), label='plambda='+"{:6.2f}".format(hRes.values["plambda"]), linewidth=2)
                if i==3:
                    axes[j,i%2].hist(timeArray,bins=tAxis,alpha=0.8)
                axes[j,i%2].set_xlabel(title[i][0])
                axes[j,i%2].set_ylabel(title[i][1])
                axes[j,i%2].set_title(ctitle[i]+" @ ch"+str(ch)) 
                axes[j,i%2].legend()

            # put fit results in arrays
            #hMean[ch].append(hRes.values["mu"])
            #hError[ch].append(hRes.errors["mu"])
            #cMean[ch].append(cRes.values["mu"])
            #cError[ch].append(cRes.errors["mu"])
            hMean[ch].append(hRes.values["plambda"])
            hError[ch].append(hRes.errors["plambda"])
            cMean[ch].append(cRes.values["plambda"])
            cError[ch].append(cRes.errors["plambda"])

            fig = plt.gcf()
            fig.set_size_inches(16.20,10.80)
            fig.tight_layout()
            fig.savefig('plotData_'+str(nfiles)+'_'+str(ch)+'.png',bbox_inches='tight',dpi=200)
            #plt.show(block=False)
       
    # draw fit results
    fig2,axes2 = plt.subplots(2,1)
    axes2[0].plot(hMean[0])
    axes2[0].plot(hMean[1])
    axes2[0].plot(hMean[2])
    axes2[0].legend()

    axes2[1].plot(cMean[0])
    axes2[1].plot(cMean[1])
    axes2[1].plot(cMean[2])
    axes2[1].legend()
    plt.show()

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    conffile=args[1]
    for i in range(len(args)-3):
        folder.append(args[i+3])
    print(len(folder))
    main()
