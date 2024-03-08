#This script is for running a check on an amplifier using Picoscope6000
#Specify signal input from picoscope into amplifier using Signal.yaml
#Then plug signal line into amp input and amp output into Ch A of picoscope
#Picoscope settings in PicoAmp.yaml - use an external function generator to trigger the Picoscope AUX port (this will trigger data collection)
#Use python3.x RunAmp.py PicoAmp.yaml

import time
import sys
import runModule as run
import daqModule as daq

config = "Example_settings.yaml"
sigFile = "SigGenSettings.yaml"

def main():
    Settings = []
    StatList = []
    print("Initialisation from config file: ",config) 
        
    Dev = []
    Daq = []
    Chan = []
    Status=[]
    SigInfo = []         

    Dev,Daq,Chan = run.load_dev(config)
    SigInfo = run.load_sig(sigFile)

    if Dev == 0: return
    Status = run.InitDAQ(Dev,Daq,Chan)
    StatList.append(Status)
    StatList.append(False)
    Settings.append([Dev,Daq,Chan])
    Nsub = Settings[0][1]["Nsubruns"]

    for n in range(Nsub):
         print("Sub Run ",n,"/",Nsub)
         StartSubRun = time.time() 
         RStats = [] 
         RStats = run.RunAmp(n,Settings,StatList,SigInfo)
         StatList = RStats
         EndSubRun = time.time()
         print("Sub Run Time = ",EndSubRun-StartSubRun)
         time.sleep(2)
     
    run.CloseDAQs(Settings,StatList) 

if __name__ == "__main__":
    print("^^^ Picoscope libraries called ^^^")
    args=sys.argv
    if(len(args)==1): #print out help if the number of arguments are wrong
       print(" ")
       print("To use DAQ, run:")
       print(" ")
       print("python3.7 RunDAQ.py YourDaqAndDeviceSettings.yaml")
       print(" ")
       print("For more details on settings:")
       print(" ")
       print("python3.7 RunDAQ.py -help")
       print(" ")
    else:
       config = args[1]  
       main()

