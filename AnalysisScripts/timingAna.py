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
from MPPCAnalysisFunctions import WfInfo,ExtractWfInfo

folder=[]
conffile="analysis_settings.yaml"

def main():
    global conffile
    nch = 0 ### Number of channel, should be constant?
    trR = 0 ### Trigger Rate
    bins=[]
    vals=[]
    #### Analyse main counter
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    myFunc.SetPolarity(waveform["Polarity"])
    myFunc.SetTimeScale(waveform["TimeScale"])
    myFunc.SetPeakThreshold(waveform["PeakThreshold"])
    myFunc.SetBins(histogram["NumberOfBins"],histogram["LowerRange"],histogram["UpperRange"])
    myFunc.SetSignalWindow(analysisWindow["Start"],analysisWindow["Stop"],analysisWindow["Baseline"])
    myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
    myFunc.EnableFFTFilter(filtering["FFTCutoffFrequency"])
    myFunc.EnableBaselineFilter()
    myFunc.SetConstantFraction(0.1)
    FList = myFunc.FileList(folder[0]+"data_1_6000")
    FileOutputs=[[],[],[],[]]  # 4channels
    SumOutputs = []
    for i in FList:
        rate = myFunc.AnalyseSingleFile(i,FileOutputs,SumOutputs)
    FList = myFunc.FileList(folder[0]+"data_2_6000")
    for i in FList:
        rate = myFunc.AnalyseSingleFile(i,FileOutputs,SumOutputs)

    conffile = "analysis_settings_3000.yaml"
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    myFunc.SetPolarity(waveform["Polarity"])
    myFunc.SetTimeScale(waveform["TimeScale"])
    myFunc.SetPeakThreshold(waveform["PeakThreshold"])
    myFunc.SetBins(histogram["NumberOfBins"],histogram["LowerRange"],histogram["UpperRange"])
    myFunc.SetSignalWindow(analysisWindow["Start"],analysisWindow["Stop"],analysisWindow["Baseline"])
    myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
    myFunc.EnableFFTFilter(filtering["FFTCutoffFrequency"])
    myFunc.EnableBaselineFilter()
    myFunc.SetConstantFraction(0.15)
    FList = myFunc.FileList(folder[0]+"data_1_3000")
    FileOutputs2=[[],[],[],[]]  # 4channels
    SumOutputs2 = []
    for i in FList:
        rate = myFunc.AnalyseSingleFile(i,FileOutputs2,SumOutputs2)
    FList = myFunc.FileList(folder[0]+"data_2_3000")
    for i in FList:
        rate = myFunc.AnalyseSingleFile(i,FileOutputs2,SumOutputs2)

    print(len(FileOutputs[0]),", ",len(FileOutputs2[0]))
    ##### Sum, Main0, Main1, Main2, Main3, Trig1, Trig2, Trig3
    dataArray=[[],[],[],[],[],[],[],[]]
    heightArray=[[],[],[],[],[],[],[],[]]
    timeArray=[[],[],[],[],[],[],[],[]]
    dataArray[0] = ExtractWfInfo(SumOutputs) #### 16ch all sum
    heightArray[0] = np.array(dataArray[0].getHeightArray(),dtype=float)
    timeArray[0]   = np.array(dataArray[0].getEdgeTimeArray(),dtype=float)
    for ch in range(4):
        print(ch)
        dataArray[ch+1]   = ExtractWfInfo(FileOutputs[ch])
        heightArray[ch+1] = np.array(dataArray[ch+1].getHeightArray(),dtype=float)
        timeArray[ch+1]   = np.array(dataArray[ch+1].getEdgeTimeArray(),dtype=float)
    for ch in range(1,4):
        print(ch+4)
        dataArray[ch+4]   = ExtractWfInfo(FileOutputs2[ch])
        heightArray[ch+4] = np.array(dataArray[ch+4].getHeightArray(),dtype=float)
        timeArray[ch+4]   = np.array(dataArray[ch+4].getEdgeTimeArray(),dtype=float)
    ret1 = np.where( (abs(timeArray[0]-950)<75) & (abs(timeArray[5]-930)<75) ) ### 1st counter has a signal
    ret2 = np.where( (abs(timeArray[0]-950)<75) & (abs(timeArray[6]-930)<75) ) ### 2st counter has a signal
    ret3 = np.where( (abs(timeArray[0]-950)<75) & (abs(timeArray[7]-930)<75) ) ### 3st counter has a signal
    timeDiff1 = (timeArray[0]-timeArray[5])[ret1]
    timeDiff2 = (timeArray[0]-timeArray[6])[ret2]
    timeDiff3 = (timeArray[0]-timeArray[7])[ret3]
    print(timeDiff1)
    print(timeDiff2)
    print(timeDiff3)
    plt.figure()
    plt.hist(timeDiff1,bins=160,range=(-80,80))
    plt.figure()
    plt.hist(timeDiff2,bins=160,range=(-80,80))
    plt.figure()
    plt.hist(timeDiff3,bins=160,range=(-80,80))
    plt.figure()
    #plt.hist2d(timeDiff1,(heightArray[5])[ret1],bins=(100,100),range=([-80,80],[0,100]))
    plt.scatter(timeDiff1,(heightArray[5])[ret1])
    plt.xlim(-80,80)
    plt.ylim(0,100)
    plt.figure()
    #plt.hist2d(timeDiff2,(heightArray[6])[ret2],bins=(100,100),range=([-80,80],[0,100]))
    plt.scatter(timeDiff2,(heightArray[6])[ret2])
    plt.xlim(-80,80)
    plt.ylim(0,100)
    plt.figure()
    #plt.hist2d(timeDiff3,(heightArray[7])[ret3],bins=(100,100),range=([-80,80],[0,100]))
    plt.scatter(timeDiff3,(heightArray[7])[ret3])
    plt.xlim(-80,80)
    plt.ylim(0,100)

    #### Pick-up the events with actual coincidence-like signals
    retFinal = np.where( (timeArray[0]>0) &
                         ( (timeArray[5]>0) & (abs(timeArray[0]-timeArray[5]-20)<10) ) |
                         ( (timeArray[6]>0) & (abs(timeArray[0]-timeArray[6]-20)<10) ) |
                         ( (timeArray[7]>0) & (abs(timeArray[0]-timeArray[7]-20)<10) )
                        )
    print("Total events : ", len(heightArray[0]))
    selected = heightArray[0][retFinal]
    print("Selected     : ", len(selected))
    plt.figure()
    plt.hist(selected,bins=180,range=(-10,890))
    plt.show()

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    conffile=args[1]
    for i in range(len(args)-2):
        folder.append(args[i+2])
    print(len(folder))
    main()
