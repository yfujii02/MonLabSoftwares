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

folder=[]

def main():
    nch = 0 ### Number of channel, should be constant?
    trR = 0 ### Trigger Rate
    bins=[]
    vals=[]
    for f in folder:
        myFunc.SetBins(115,-5,109)
        myFunc.SetSignalWindow(195,275,175)
        myFunc.EnableMovingAverageFilter()
        myFunc.EnableFFTFilter()
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
    plt.show()

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    for i in range(len(args)-1):
        folder.append(args[i+1])
    print(len(folder))
    main()
