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
    print("                     1: ChA&ChB coincidence for negative signal")
    print("                     2: ChA&ChB coincidence for positive signal")
    print("     [4] : Output file name                  ")
    print("     [5] : Threshold in mV                   ")
    exit()
print('Number of events per each sub run:',args[1])
print('Number of subruns for each run   :',args[2])
print('DAQ mode                         :',args[3])
print("                     0: pedestal trigger     ")
print("                     1: ChA&ChB coincidence for negative signal")
print("                     2: ChA&ChB coincidence for positive signal")
print('Output file name w/o ".XXX"      :',args[4])
print('Threshold in mV                  :',args[5])

num_events=int(args[1])
num_subruns=int(args[2])
if num_subruns<1 or num_events<1:
    print("You're wrong!")
    exit()
trig=int(args[3])
if trig<0 or trig>2:
    print("Invalid DAQ mode ", trig)
    print("                     0: pedestal trigger     ")
    print("                     1: ChA&ChB coincidence for negative signal")
    print("                     2: ChA&ChB coincidence for positive signal")
    exit()

fname=args[4]
threshold=float(args[5]) # mV
if(threshold<0):
    threshold=-1*threshold

#### Set parameters
# Num of events per sub run
# Thresholds in mV
# DAQ mode (pedestal or selftrigger)
myTestDAQ.set_params(num_events, threshold, trig, fname)
myTestDAQ.init_daq()
for i in range(num_subruns):
    print('Sub run: ',i,'/',num_subruns)
    myTestDAQ.run_daq(i)
    time.sleep(1)
myTestDAQ.close()

