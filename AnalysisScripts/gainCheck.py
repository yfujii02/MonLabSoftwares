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
from fitFunctions import SPPeaksGaus4

folder=[]

def main():
    nch = 0 ### Number of channel, should be constant?
    trR = 0 ### Trigger Rate
    bins=[]
    vals=[]
    for f in folder:
        myFunc.SetRMSCut(1.25)
        myFunc.SetBins(228,-5,109)
        myFunc.SetSignalWindow(195,280,175) ## Signal window [start,stop] + Baseline end point
        myFunc.EnableMovingAverageFilter(24)  ## Also set Number of averagint points
        myFunc.EnableFFTFilter(396)         ## Cut-off frequency in MHz
        myFunc.EnableBaselineFilter()
        nch, trR, hData = myFunc.AnalyseFolder(f,False)
        #plt.show()
        print(len(hData))
        if len(bins)==0:
            for i in range(int(len(hData)/2)):
                bins.append(hData[2*i])
                vals.append(hData[2*i+1])
        else:
            for i in range(int(len(hData)/2)):
                vals[i] = vals[i]+hData[2*i+1]
        plt.close('all')
    errFunc=[[],[],[],[],[]]
    for i in range(len(bins)-1):
        guess = [1024,512,256,128,2,2,2,2,0,4.1]
        xdata = bins[i][:-1][bins[i][:-1]<15.5]
        ydata = vals[i][bins[i][:-1]<15.5]
        popt,pcov = curve_fit(SPPeaksGaus4, xdata=xdata, ydata=ydata, p0=guess, maxfev=5000,
                              bounds = ((1,1,1,1, 0.1, 0.1, 0.1, 0.1, -1, 0.1),
                                        (np.inf,np.inf,np.inf,np.inf, 4,4,4,4, 1, 5)) )
        perr = np.sqrt(np.diag(pcov))
        print("Single p.e. = %.2f +/- %.2f [mV]" % (popt[9],perr[9]))
        plt.figure()
        plt.bar(bins[i][:-1],vals[i],width=bins[i][1]-bins[i][0],color='blue')
        xval = np.linspace(-0.5*popt[9]+popt[8],popt[8]+3.5*popt[9],40)
        plt.plot(xval,SPPeaksGaus4(xval, *popt),c='r',label='4 Gaus')
        plt.xlim(-4,36)
        plt.ylim(0.2,popt[0]*1.2)
        plt.text(15,popt[0]*0.4,str(r'1p.e.=%.2f $\pm$ %.2f' % (popt[9],perr[9])),fontsize=18)
        plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    for i in range(len(args)-1):
        folder.append(args[i+1])
    print(len(folder))
    main()
