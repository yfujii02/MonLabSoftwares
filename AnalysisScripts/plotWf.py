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

files=[]
conffile="analysis_settings.yaml"

def main(output,plotch,start,end):
    nch = 0 ### Number of channel, should be constant?
    trR = 0 ### Trigger Rate
    bins=[]
    vals=[]
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    count=0
    for file in files:
        myFunc.SetPolarity(waveform["Polarity"])
        myFunc.SetTimeScale(waveform["TimeScale"])
        myFunc.SetPeakThreshold(waveform["PeakThreshold"])
        myFunc.SetBins(histogram["NumberOfBins"],histogram["LowerRange"],histogram["UpperRange"])
        myFunc.SetSignalWindow(analysisWindow["Start"],analysisWindow["Stop"],analysisWindow["Baseline"])
        myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
        myFunc.EnableFFTFilter(filtering["FFTCutoffFrequency"])
        myFunc.EnableBaselineFilter()
        nch, trR, hData = myFunc.AnalyseFolder(f,False)
        #plt.show()
        myFunc.PlotWaveformsFromAFile(file,plotch,start,end) 
        #plt.close('all')
        plt.show()
        plt.savefig(f'{output}_{count}.png')
        count=count+1

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    conffile=args[1]
    output='wf_image'
    start=end=-1
    plotch=-1
    for i in range(len(args)-2):
        if args[i+2]=='-fig':
            output=args[i+3]
            i=i+1
        if args[i+2]=='-start':
            start=int(args[i+3])
            i=i+1
        if args[i+2]=='-end':
            end=int(args[i+3])
            i=i+1
        if args[i+2]=='-ch':
            plotch=int(args[i+3])
            i=i+1
        files.append(args[i+2])
    main(output,plotch,start,end)
