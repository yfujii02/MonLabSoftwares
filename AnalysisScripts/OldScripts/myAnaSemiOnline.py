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
start=0
end=0
sigTiming = 1550
deltaT    =   50

def main():
    nch = 0 ### Number of channel, should be constant?
    Nevents = 0
    NumCoin = 0 # Number of coincidence events with given time difference in ChX and ChY
    bins=[]
    vals=[]
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    myFunc.SetPolarity(waveform["Polarity"])
    myFunc.SetTimeScale(waveform["TimeScale"])
    myFunc.SetPeakThreshold(waveform["PeakThreshold"])
    myFunc.SetBins(histogram["NumberOfBins"],histogram["LowerRange"],histogram["UpperRange"])
    myFunc.SetSignalWindow(analysisWindow["Start"],analysisWindow["Stop"],analysisWindow["Baseline"])
    myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
    myFunc.EnableFFTFilter(filtering["FFTCutoffFrequency"])
    myFunc.EnableBaselineFilter()
    FList = myFunc.FileList(folder[0])
    FileOutputs=[[],[],[],[]]  # 4channels
    SumOutputs = []
    for i in range(start,end):
        myFunc.AnalyseSingleFile(FList[i],FileOutputs,SumOutputs)
    Nevents = myFunc.GetNumEvents()
    #nch,trR,hData,Nevents = myFunc.AnalyseFolder(folder[0],False,start,end)
    dataArray=[[],[],[],[]]
    heightArray=[[],[],[],[]]
    chargeArray=[[],[],[],[]]
    timeArray=[[],[],[],[]]
    histData=[]
    RangeLower = myFunc.RangeLower
    RangeUpper = myFunc.RangeUpper
    NBins      = myFunc.NBins
    SigLower   = myFunc.SigLower
    SigUpper   = myFunc.SigUpper
    TimeScale  = myFunc.TimeScale
    TimeLower  = myFunc.TimeLower
    TimeUpper  = myFunc.TimeUpper
    TimeBins   = myFunc.TimeBins
    for ch in range(4):
        print(ch)
        dataArray[ch]   = myFunc.ExtractWfInfo(FileOutputs[ch])
        heightArray[ch] = np.array(dataArray[ch].getHeightArray(),dtype=float)
        chargeArray[ch] = np.array(dataArray[ch].getChargeArray(),dtype=float)
        timeArray[ch]   = np.array(dataArray[ch].getEdgeTimeArray(),dtype=float)
        ##### Checking if they have pulses near the accelerator timing
        nBins, vals = myFunc.PlotHistogram(heightArray[ch],RangeLower,RangeUpper,NBins,str(dataArray[ch].getChannel(0)),
                "Peak height [mV]")
        histData.append(nBins)
        histData.append(vals)
        nBins, vals = myFunc.PlotHistogram(chargeArray[ch],RangeLower,myFunc.RangeUpper*TimeScale*(SigUpper-SigLower)/4.0,NBins,str(dataArray[ch].getChannel(0)),
                "Charge [mV*ns]")
        histData.append(nBins)
        histData.append(vals)
        nBinsT, valsT = myFunc.PlotHistogram(timeArray[ch],TimeScale*TimeLower,TimeScale*TimeUpper,TimeBins,str(dataArray[ch].getChannel(0)),
                "Edge Time (ns)")
        histData.append(nBinsT)
        histData.append(valsT)        
    histData = np.array(histData)
    ##### Ch2 = 1s Main counter and compare timing with others
    ret0 = np.where( (abs(timeArray[0]-sigTiming)<deltaT) & (abs(timeArray[1]-sigTiming)<deltaT) ) ##### Should have the same time
    ret1 = np.where( (abs(timeArray[2]-sigTiming)<deltaT) & (abs(timeArray[0]-sigTiming)<deltaT) )
    ret2 = np.where( (abs(timeArray[2]-sigTiming)<deltaT) & (abs(timeArray[1]-sigTiming)<deltaT) )
    ret3 = np.where( (abs(timeArray[2]-sigTiming)<deltaT) & (abs(timeArray[3]-sigTiming)<deltaT) ) ##### This one is the most reliable (or the important parameter?)
    timeDiffCounters0 = (timeArray[0]-timeArray[1])[ret0] #### 2nd counter Ch B and Ch D
    timeDiffCounters1 = (timeArray[2]-timeArray[0])[ret1] #### Btw 2nd counter Ch B
    timeDiffCounters2 = (timeArray[2]-timeArray[1])[ret2] #### Btw 2nd counter Ch D
    timeDiffCounters3 = (timeArray[2]-timeArray[3])[ret3] #### Btw Finger counter
    NumCoin = len(timeDiffCounters3)
    bins=[]
    vals=[]
    if len(bins)==0:
        for i in range(int(len(histData)/2)):
            bins.append(histData[2*i])
            vals.append(histData[2*i+1])
    else:
        for i in range(int(len(histData)/2)):
            vals[i] = vals[i]+histData[2*i+1]

    plt.close()
    DataTypes = ['Peak Voltage','Int Charge','Timing'] 
    Colours = ['blue','green','red']
    Units = ['Voltage (mV)', 'Charge (mV ns)', 'Time (ns)']
    fig = plt.figure(constrained_layout=True)

    subfigs = fig.subfigures(1, 3)
    #print(bins)
    for outerind, subfig in enumerate(subfigs.flat):
        subfig.suptitle(f'{DataTypes[outerind]}')
        axs = subfig.subplots(2, 2)
        for innerind, ax in enumerate(axs.flat):
            dataIndex = outerind+innerind*len(DataTypes)
            #0 0 0 1 1 1 2 2 2 3 3 3
            #0 1 2 3 0 1 2 3 0 1 2 3
            ax.bar(bins[dataIndex][:-1],vals[dataIndex],width=bins[dataIndex][1]-bins[dataIndex][0],color=Colours[outerind])
            ax.set_title(f'Ch = {innerind}', fontsize='small')
            if(innerind == 2 or innerind == 3): ax.set_xlabel(f'{Units[outerind]}',fontsize='small')
    fig = plt.figure()
    timeStart = 0.8*analysisWindow["Start"]
    timeStop  = 0.8*analysisWindow["Stop"] 
    plt.hist(timeArray[0],range=[timeStart,timeStop],bins=100,alpha=0.8)
    plt.hist(timeArray[1],range=[timeStart,timeStop],bins=100,alpha=0.8)
    plt.hist(timeArray[2],range=[timeStart,timeStop],bins=100,alpha=0.8)
    plt.hist(timeArray[3],range=[timeStart,timeStop],bins=100,alpha=0.8)
    fig = plt.figure()
    plt.hist(timeDiffCounters0,range=[-50,50],bins=100,alpha=0.8)
    plt.hist(timeDiffCounters1,range=[-50,50],bins=100,alpha=0.8)
    plt.hist(timeDiffCounters2,range=[-50,50],bins=100,alpha=0.8)
    plt.hist(timeDiffCounters3,range=[-50,50],bins=100,alpha=0.8)

    print(Nevents)        
    print("Efficiency = ",float(NumCoin/Nevents))
    plt.show()
        

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    conffile=args[1]
    start=int(args[2])
    end=int(args[3])
    sigTiming=int(args[4]) #### Expected signal timing w.r.t the trigger timing
    if (len(args)-5>1):
        sys.exit("Only one folder can be analysed")
    folder.append(args[5])
    main()
