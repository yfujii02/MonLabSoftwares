import time
from multiprocessing import Process
from threading import Thread
import sys
import yaml
import daqModule as daq
import daqModule0 as daq0
import daqModule1 as daq1
import daqModule2 as daq2

devFile = "Example_settings.yaml"
InitFlag = False

def load_dev(config_file):
    #Read config file and return device and daq settings
    with open(config_file,"r") as f:
        config = yaml.safe_load(f)
        DeviceSettings = config[0]["DeviceSettings"]
        DaqSettings = config[1]["DaqSettings"]
        ChannelSettings = config[2]["ChannelSettings"]

    #DeviceStatus = [DeviceSettings["typeDev"],DeviceSettings["numDev"],DaqSettings["Nsubruns"],False,False]

    return DeviceSettings, DaqSettings, ChannelSettings

def InitDAQ(DeviceInfo,DAQSettings,ChannelSettings):
    #Print help on settings to terminal
    #if(devFile=="Example_settings.yaml" or devFile =='-help'):
    #    fileEx = open('Example_settings.yaml')
    #    print("Contents of example settings file (Example_settings.yaml):")
    #    print("##################################")
    #    print(" ")
    #    for line in fileEx:
    #        print(line)
    #    print("##################################")
    #    print(" ")
    #    return
    
    #Load settings for device and daq from .yaml file
    #print("Initial, ",time.time())
    #DeviceInfo, DAQSettings, ChannelSettings = Load(devFile)
    #print("After load settings file, ",time.time())
   
    #daq.set_params(DeviceInfo,DAQSettings,ChannelSettings) #send parameters to daq functions
    #print("After set_params(), ",time.time())
    Stat = daq.init_daq(DeviceInfo,DAQSettings,ChannelSettings) #initialise daq with specified parameters
    #Stat=[]
    #p = Process(target=daq.init_daq,args=(DeviceInfo,DAQSettings,ChannelSettings,Stat)) #initialise daq with specified parameters
    #p.start()
    #time.sleep(2)
    #p.join()
     
    return Stat

def RunDAQ(SubRun,Settings,Stat):
    #StartProgramTime=time.time() #record start of program time
    #print("Start time = ",StartProgramTime) 
    
    #Collect specified number of triggers a total of nSub times by running run_daq function 
    Modules = [daq0.run_daq,daq1.run_daq,daq2.run_daq] #Add more for more devices...
    Processes = [None]*len(Settings)
    RetStats=[]
    for j in range(len(Settings)): 
        RetStats.append(Stat[2*j])
        RetStats.append(False)
    print("Number of devices: ",len(Settings))
    for i in range(len(Settings)):
        
        print("SubRun ",i)
        print(Stat[i])
        print(Settings[i])
        print("Device #",i)
        #if(Settings[i][0]['typeDev'] == '3000'):
        
        #    Processes[i] = Thread(target = daq3.run_daq, args = (SubRun,Settings[i],Stat[2*i],RetStats,2*i))

        #elif(Settings[i][0]['typeDev']=='6000'):
        
        #    Processes[i] = Thread(target = daq6.run_daq, args = (SubRun,Settings[i],Stat[2*i],RetStats,2*i))

        Processes[i] = Thread(target = Modules[i], args = (SubRun,Settings[i],Stat[2*i],RetStats,2*i))
        Processes[i].start()
        
    
    #for p in Processes: p.join()
    time.sleep(2)
    Check=False
    while(Check  == False):
        Check = True
        for l in range(int(len(RetStats)/2)):
            if(RetStats[l+1]==False):Check=False
   
    for p in Processes: p.join()
   
 
    return RetStats
def CloseDAQs(Settings,Stat): 
    for i in range(len(Settings)): daq.close(Settings[i],Stat[2*i]) #close the daq
    return

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

