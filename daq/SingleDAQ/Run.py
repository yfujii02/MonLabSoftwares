import time
from multiprocessing import Process
import sys
import numpy as np
import runModule as run
import daqModule as daq
import threading

devFile = "Example_settings.yaml"

def main():
    Settings = []
    StatList = []
    n = 0
    print("Initialisation from config file: ",devFile) 
        
    Dev = []
    Daq = []
    Chan = []
    Status=[]         

    Dev,Daq,Chan = run.load_dev(devFile)
    if Dev == 0: return
    Status = run.InitDAQ(Dev,Daq,Chan)
    Settings.append([Dev,Daq,Chan])
    Nsub = Settings[0][1]["Nsubruns"]

    for n in range(Nsub):
         print("Sub Run ",n,"/",Nsub)
         StartSubRun = time.time() 
         #print('Sub run inputs')
         #print('Settings')
         #print(Settings)
         #print('StatList')
         #print(StatList) 
         run.RunDAQ(n,Settings)
         EndSubRun = time.time()
         print("Sub Run Time = ",EndSubRun-StartSubRun)
         time.sleep(2)
     
    run.CloseDAQs(Settings) 

if __name__ == "__main__":
    print("^^^ Picoscope libraries called ^^^")
    args=sys.argv
    if(len(args)!=2): #print out help if the number of arguments are wrong
       print(" ")
       print("To use DAQ, run:")
       print(" ")
       print("python3 RunDAQ.py YourDaqSettings.yaml")
       print(" ")
       print("For more details on settings:")
       print(" ")
       print("python3 RunDAQ.py -help")
       print(" ")
    else:
        devFile = args[1]
       
        main()

