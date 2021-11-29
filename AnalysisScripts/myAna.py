import sys
import numpy as np
import MPPCAnalysisFunctions as myFunc
import matplotlib.pyplot as plt

folder=""

def main():
    myFunc.AnalyseFolder(folder,False)
    plt.show()

if __name__ == "__main__":
    args=sys.argv
    #### name of the folder where you have the data to be analysed
    folder=args[1]
    main()
