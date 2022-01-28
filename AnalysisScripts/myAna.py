import sys
import numpy as np
import MPPCAnalysisFunctions as myFunc
import matplotlib.pyplot as plt

folder=""

def main():
    # folder=r'C:\Users\BemusedPopsicle\Desktop\Uni Stuff\ScinData\2021-11-*\\'
    # myFunc.AnalyseFolder(folder,False)

    myFunc.CosmicSr90Analysis("Near Jacketed",Sr=0,Cosmic=1)

    # s = 0
    # folder=r'C:\Users\BemusedPopsicle\Desktop\Uni Stuff\ScinData\\'
    # file=folder + r'2021-12-15-Far-Cosmic\data10.npy'
    # myFunc.PlotWaveformsFromAFile(file,fn=myFunc.FullAnalysis,SingleWf=s,title="Far ")
    plt.show()

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    if len(args) > 1: folder = args[1]
    # folder=r'C:\Users\BemusedPopsicle\Desktop\Uni Stuff\ScinData\\'
    main()
