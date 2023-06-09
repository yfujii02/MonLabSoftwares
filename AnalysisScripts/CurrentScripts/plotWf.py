#######################################################
#
# This is a minimum example script to run the analysis
# calling function to make a charge distribution plot
#
#######################################################

import sys
import numpy as np
import MPPCAnalysisFunctions23 as myFunc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from scipy.optimize import curve_fit
from fitFunctions import Moyal,Gaus
from loadSettings import load_analysis_conf
import os

files=[]
conffile="analysis_settings.yaml"

def main(output,plotch,start,end):
    nch = 0 ### Number of channel, should be constant?
    trR = 0 ### Trigger Rate
    bins=[]
    vals=[]
    waveform,analysisWindow,filtering,histogram=load_analysis_conf(conffile)
    count=0
    print(files)
    for file in files:
        myFunc.SetPolarity(waveform["Polarity"])
        myFunc.SetTimeScale(waveform["TimeScale"])
        myFunc.SetPeakThreshold(waveform["PeakThreshold"])
        myFunc.SetBins(histogram["NumberOfBins"],histogram["LowerRange"],histogram["UpperRange"])
        myFunc.SetSignalWindow(np.array(analysisWindow["Start"]),np.array(analysisWindow["Stop"]),np.array(analysisWindow["Baseline"]))
        myFunc.EnableMovingAverageFilter(filtering["MovingAveragePoints"])
        myFunc.EnableFFTFilter(filtering["UpperFFTCutoffFrequency"],filtering["LowerFFTCutoffFrequency"])
       # myFunc.EnableBaselineFilter()
        #nch, trR, hData = myFunc.AnalyseFolder(f,False)
        #plt.show()
        FileString = folder+str(file)+ext
        fname = 'data_'+str(file)
        myFunc.PlotWaveformsFromAFile(FileString,plotch,start,end) 
        #plt.close('all')
       # plt.show()
        plt.savefig(f'{output}_{count}.png')
        count=count+1
   # os.system("mv ./"+output+"*.png /mnt/c/Users/yfuj0004/work/WaveformPlots/")

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    if(len(args)==1):
        print("python plotWf.py ConfigFile ") 
    conffile=args[1]
    output='wf_image'
    start=end=-1
    plotch=-1
    Skip=0
    folder ='./'
    ext = '.npy'
    for i in range(len(args)-2):
        print(i,' ',args[i+2])
        if(Skip == 1):
            Skip=0
        else:
            if args[i+2]=='fig':
                output=str(args[i+3])
                Skip=1
                print(f'output name = {output}')
            elif args[i+2]=='start':
                start=int(args[i+3])
                Skip=1
            elif args[i+2]=='end':
                end=int(args[i+3])
                Skip=1
            elif args[i+2]=='ch':
                plotch=int(args[i+3])
                Skip=1
            elif args[i+2]=='folder':
                folder = str(args[i+3])
                Skip=1
            elif args[i+2]=='ext':
                ext = str(args[i+3])
                Skip=1
            else:
                files.append(args[i+2])
    main(output,plotch,start,end)
