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
from fitFunctions import Moyal

folder=[]

def main():
    nch = 0 ### Number of channel, should be constant?
    trR = 0 ### Trigger Rate
    bins=[]
    vals=[]
    for f in folder:
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
    for i in range(len(bins)):
        plt.figure()
        #print(len(bins[i]),len(vals[i]))
        #print(bins[i])
        #print(vals[i])
        plt.bar(bins[i][:-1],vals[i],width=bins[i][1]-bins[i][0],color='blue')
        plt.yscale('log')
        if i==nch:
            ### Try Laudau fitting
            xdata = bins[i][:-1][bins[i][:-1]>100]
            ydata = vals[i][bins[i][:-1]>100]
            ydata = ydata[xdata<360]
            xdata = xdata[xdata<360]
            pini  = [1000,175,1]
            popt,pcov =  curve_fit(Moyal, xdata=xdata,ydata=ydata,p0=pini,maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            print(popt)
            print(perr)
            plt.plot(xdata,Moyal(xdata,*popt),c='r',label='Landau')
    plt.show()

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    for i in range(len(args)-1):
        folder.append(args[i+1])
    print(len(folder))
    main()
