import time
import ctypes
from multiprocessing import Process
from threading import Thread
import sys
import yaml
import daqModule as daq

devFile = "Example_settings.yaml"
InitFlag = False

def load_dev(config_file):
    #Read config file and return device and daq settings
    if(config_file =='-help'):
        fileEx = open('Example_settings.yaml')
        print("Contents of example settings file (Example_settings.yaml):")
        print("##################################")
        print(" ")
        for line in fileEx:
            print(line)
        print("##################################")
        print(" ")
        return 0,0,0
    else:
        with open(config_file,"r") as f:
            config = yaml.safe_load(f)
            DeviceSettings = config[0]["DeviceSettings"]
            DaqSettings = config[1]["DaqSettings"]
            ChannelSettings = config[2]["ChannelSettings"]
        return DeviceSettings, DaqSettings, ChannelSettings

def load_sig(sig_file):
    if(sig_file=="Example_settings.yaml" or sig_file =='-help'):
        fileEx = open('Example_settings.yaml')
        print("Contents of example settings file (Example_settings.yaml):")
        print("##################################")
        print(" ")
        for line in fileEx:
            print(line)
        print("##################################")
        print(" ")
        return 0,0,0
    else:
        with open(sig_file,"r") as f:
            sig = yaml.safe_load(f)
            SigSettings = sig[0]["SigSettings"]
        return SigSettings

def InitDAQ(DeviceInfo,DAQSettings,ChannelSettings):
    cHandle = ctypes.c_int16()
    Status  = {}
    daq.init_daq(DeviceInfo,DAQSettings,ChannelSettings,Status,cHandle) #initialise daq with specified parameters

def RunDAQ(SubRun,Settings):
    #Collect specified number of triggers a total of nSub times by running run_daq function     
    daq.run_daq(SubRun,Settings)

def CloseDAQs(Settings): 
    daq.close(Settings) #close the daq
    return

def main():
    Settings = []
    StatList = []
    print("Initialisation from config file: ",devFile) 
    Dev = []
    Daq = []
    Chan = []
    Status=[]

    Dev,Daq,Chan = load_dev(devFile)
    if Dev == 0: return
    Status = InitDAQ(Dev,Daq,Chan)
    StatList.append(Status)
    StatList.append(False)
    Settings= [Dev,Daq,Chan]
    
    Nsub = Settings[1]["Nsubruns"]

    for n in range(Nsub):
         print("Sub Run ",n,"/",Nsub)
         StartSubRun = time.time() 
         #print('Sub run inputs')
         #print('Settings')
         #print(Settings)
         #print('StatList')
         #print(StatList)
         RStats = [] 
         RStats = RunDAQ(n,Settings)
         StatList = RStats
         EndSubRun = time.time()
         print("Sub Run Time = ",EndSubRun-StartSubRun)
         time.sleep(1)
     
    CloseDAQs(Settings) 

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

