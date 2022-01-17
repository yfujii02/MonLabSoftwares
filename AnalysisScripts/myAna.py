import sys
import numpy as np
import MPPCAnalysisFunctions as myFunc
import matplotlib.pyplot as plt

folder=""

def main():
    # folder=r'C:\Users\BemusedPopsicle\Desktop\Uni Stuff\ScinData\2021-11-*\\'
    # myFunc.AnalyseFolder(folder,False)
    myFunc.CosmicSr90Analysis("Middle",Sr=0,Cosmic=1)
    # s = 0
    # folder=r'C:\Users\BemusedPopsicle\Desktop\Uni Stuff\ScinData\\'
    # file = folder + r'2021-11-29-Strontium\ScintTestNov29_Sr90_AUXT35mV_2_35.npy'
    # myFunc.PlotWaveformsFromAFile(file,fn=myFunc.FullAnalysis,SingleWf=s,title="Full Analysis ")
    # myFunc.PlotWaveformsFromAFile(file,fn=myFunc.NoBaseline,SingleWf=s,title="No Baseline ")
    # myFunc.PlotWaveformsFromAFile(file,SingleWf=s,title="Unprocessed ")
    #b = myFunc.baselines
    #print(np.mean(b),np.std(b))
    #myFunc.PlotHistogram(b,np.max(b),np.min(b),30,'Baselines ({})'.format(len(b)),'Voltage [mv]')
    plt.show()

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    if len(args) > 1: folder = args[1]
    # folder=r'C:\Users\BemusedPopsicle\Desktop\Uni Stuff\ScinData\\'
    main()
