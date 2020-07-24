import time
import sys
import myTestDAQ

args=sys.argv
if len(args)!=6:
    print("Wrong number of arguments!!")
    print("Usage :\n python3 ",args[0]," [1] [2] [3]")
    print("     [1] : Number of events per each sub run ")
    print("     [2] : Number of subruns for each run    ")
    print("     [3] : DAQ mode  0: pedestal trigger     ")
    print("                     1: Two hits coincidence for negative signal")
    print("                     2: Two hits coincidence for positive signal")
    print("     [4] : Output file name                  ")
    print("     [5] : Threshold in mV                   ")
    exit()
print('Number of events per each sub run:',args[1])
print('Number of subruns for each run   :',args[2])
print('DAQ mode                         :',args[3])
print("                     0: pedestal trigger     ")
print("                     1: Two hits coincidence for negative signal")
print("                     2: Two hits coincidence for positive signal")
print('Output file name w/o ".XXX"      :',args[4])
print('Threshold in mV                  :',args[5])

num_events=int(args[1])
num_subruns=int(args[2])
if num_subruns<1 or num_events<1:
    print("You're wrong!")
    exit()
daqMode=int(args[3])
if daqMode<0 or daqMode>2:
    print("Invalid DAQ mode ", daqMode)
    print("                     0: pedestal trigger     ")
    print("                     1: Two hits coincidence for negative signal")
    print("                     2: Two hits coincidence for positive signal")
    exit()

fname=args[4]
threshold=float(args[5]) # mV
if(threshold<0):
    threshold=-1*threshold

#### Set parameters
# Num of events per sub run
# Thresholds in mV
# DAQ mode (pedestal or selftrigger)
readchannel="1111" # Read    channels for ABCD. Corresponding channel is read if it's not zero (1)
#trigchannel="1100" # Trigger channels for ABCD. Corresponding channel is used in trigger if it's not zero (1)
trigchannel="0011" # Trigger channels for ABCD. Corresponding channel is used in trigger if it's not zero (1)
myTestDAQ.set_params(num_events, threshold, daqMode, fname, readchannel, trigchannel)
myTestDAQ.init_daq()

for i in range(num_subruns):
    print('Sub run: ',i,'/',num_subruns)
    Start = time.time()
    myTestDAQ.run_daq(i)
    End = time.time()
    eTime = End-Start
    print("RunDaq Time = ", eTime) 
    TimeOutFlag=myTestDAQ.getTimeOutFlag()
    if(TimeOutFlag==True):
        print("Resetting DAQ...") 
        TimeOutFlag==False
        myTestDAQ.close()
        time.sleep(1)
        myTestDAQ.set_params(num_events, threshold, daqMode, fname, readchannel, trigchannel)
        myTestDAQ.init_daq()
    time.sleep(1)
myTestDAQ.close()

