import time
import sys
import runModule as run
import daqModule as daq

devFile1 = "Example_settings.yaml"
devFile2 = "Example_settings.yaml"

devFiles = []

def main():
    Settings = []
    StatList = []
    n = 0
    for config in devFiles:
        print("Initialisation from config file: ",config) 
        
        Dev = []
        Daq = []
        Chan = []
        Status=[]         

        Dev,Daq,Chan = run.load_dev(config)
        if Dev == 0: return
        Status = run.InitDAQ(Dev,Daq,Chan)
        StatList.append(Status)
        StatList.append(False)
        Settings.append([Dev,Daq,Chan])
        if n>0: 
            if Daq["Nsubruns"]!=Settings[0][1]["Nsubruns"]: 
                print("Warning: Mismatch in subruns - will use first device subrun settings")
        n+=1
    
    Nsub = Settings[0][1]["Nsubruns"]

    for n in range(Nsub):
         print("Sub Run ",n,"/",Nsub)
         StartSubRun = time.time() 
         #print('Sub run inputs')
         #print('Settings')
         #print(Settings)
         #print('StatList')
         #print(StatList)
         RStats = [] 
         RStats = run.RunDAQ(n,Settings,StatList)
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
       
       for i in range(len(args)-1):
          devFiles.append(args[i+1])
       
       main()

