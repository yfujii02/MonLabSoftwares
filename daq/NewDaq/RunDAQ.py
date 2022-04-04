import time
import sys
import daqModule as daq
from loadSettings import load_dev as Load

devFile = "Example_settings.yaml"

def main():
    #Print help on settings to terminal
    if(devFile=="Example_settings.yaml" or devFile =='-help'):
        fileEx = open('Example_settings.yaml')
        print("Contents of example settings file (Example_settings.yaml):")
        print("##################################")
        print(" ")
        for line in fileEx:
            print(line)
        print("##################################")
        print(" ")
        return
    
    #Load settings for device and daq from .yaml file
    print("Initial, ",time.time())
    DeviceInfo, DAQSettings, ChannelSettings = Load(devFile)
    print("After load settings file, ",time.time())
    nSub = DAQSettings["Nsubruns"] #read number of subruns to perform
    daq.set_params(DeviceInfo,DAQSettings,ChannelSettings) #send parameters to daq functions
    print("After set_params(), ",time.time())
    daq.init_daq() #initialise daq with specified parameters
  
    StartProgramTime=time.time() #record start of program time
    print("Start time = ",StartProgramTime) 
    
    #Collect specified number of triggers a total of nSub times by running run_daq function 
    for i in range(nSub): 
        print("Sub run: ",i,"/",nSub)
        Start = time.time()
        daq.run_daq(i)
        End = time.time()
        print("Run DAQ time = ",End-Start)
    daq.close() #close the daq

if __name__ == "__main__":
    print("^^^ Picoscope libraries called ^^^")
    args=sys.argv
    if(len(args)!=2): #print out help if the number of arguments are wrong
       print(" ")
       print("To use DAQ, run:")
       print(" ")
       print("python3.7 RunDAQ.py YourDAQandDeviceSettings.yaml")
       print(" ")
       print("For more details on settings:")
       print(" ")
       print("python3.7 RunDAQ.py -help")
       print(" ")
    else:   
       devFile = args[1] #read in specified settings file
       main() 

