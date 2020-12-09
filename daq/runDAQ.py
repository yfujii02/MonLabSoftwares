import time
import sys
import myTestDAQ
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

### global parameters
num_events=100
num_subruns=1
daqMode=3
fname='fileName'
threshold=100

#### Set parameters
# Num of events per sub run
# Thresholds in mV
# DAQ mode (pedestal or selftrigger)
#readchannel="1111" # Read    channels for ABCD. Corresponding channel is read if it's not zero (1)
readchannel="1100" # Trigger channels for ABCD. Corresponding channel is used in trigger if it's not zero (1)
#trigchannel="0011" # Trigger channels for ABCD. Corresponding channel is used in trigger if it's not zero (1)
trigchannel="0100" # Trigger channels for ABCD. Corresponding channel is used in trigger if it's not zero (1)

#genPulseV=1.000*2.0  # in V used for 385nm
#genPulseV=0.870*2.0  # in V used for 405nm
#genPulseV=0.883*2.0  # in V used for 405nm (5/Oct/2020; from plastic fibre meas.-)
#genPulseV=0.850*2.0  # in V used for 470nm
#genPulseV=0.860*2.0  # in V used for 470nm (6/Oct/2020)
#genPulseV=0.775*2.0  # in V used for 525nm
#genPulseV=0.780*2.0  # in V used for 525nm (6/Oct/2020)
#genPulseV=0.360*2.0  # in V used for 585nm
#genPulseV=0.220*2.0  # in V used for 626nm
#genPulseV=0.225*2.0  # in V used for 626nm (8/Oct/2020)
#genPulseV=0.266*2.0 # in V used for 626 nm (10/11/2020)
#genPulseV=0.260*2.0 #in V used for 626 nm (11/11/2020)
genPulseV=0.7 #in V used for 626 nm (12/11/20 now with 50 Ohm Splitter)

genPulseRate=20  # in MHz
# Voltage Ranges for channel ABCD
# 2 = 50 mV
# 3 = 100 mV 
# 4 = 200 mV
# 5 = 500 mV
# 6 = 1 V
# 7 = 2 V
# 8 = 5 V 
# 9 = 10 V 
volRanges="6733"

plotEachFig=False

def setPlotEachFig(val):
    global plotEachFig
    plotEachFig = val
    return val

def initDAQ(mode):
    myTestDAQ.set_params(num_events, threshold, daqMode, fname, readchannel, trigchannel, volRanges)
    if mode==3:
        myTestDAQ.set_pulseParam(int(genPulseV*1e6), int(genPulseRate*1e6))
    myTestDAQ.init_daq()
    return True

## If you want to call the main function from jupyter-lab/jupyter-notebook or other python,
## set parameters with this function!!
def setParameters(val0,val1,val2,val3,val4):
    global num_subruns
    global num_events
    global threshold
    global daqMode
    global fname
    global readchannel
    global trigchannel
    global volRanges
    num_subruns=int(val0)
    num_events =int(val1)
    threshold  =float(val2)
    daqMode    =int(val3)
    fname      =val4
    readchannel="1100" # Temporary hardcoded
    trigchannel="0100" # Temporary hardcoded
    volRanges  ="6733" # Temporary hardcoded
    setPlotEachFig(True)

def main():
    initDAQ(daqMode)
    
    for i in range(num_subruns):
        print('Sub run: ',i,'/',num_subruns)
        Start = time.time()
        myTestDAQ.run_daq(i,0) #26/11 added run number to run_daq() - set to 0 when not required
        End = time.time()
        eTime = End-Start
        print("RunDaq Time = ", eTime)
        if plotEachFig==True:
            img = mpimg.imread('/home/comet/Desktop/figA.png')
            imgplot = plt.imshow(img)
            plt.show()
        TimeOutFlag=myTestDAQ.getTimeOutFlag()
        if(TimeOutFlag==True):
            print("Resetting DAQ...") 
            TimeOutFlag==False
            myTestDAQ.close()
            time.sleep(1)
            initDAQ()
        time.sleep(1)
    myTestDAQ.close()

if __name__ == "__main__":
    args=sys.argv
    if len(args)!=6:
        print("Wrong number of arguments!!")
        print("Usage :\n python3 ",args[0]," [1] [2] [3]")
        print("     [1] : Number of events per each sub run ")
        print("     [2] : Number of subruns for each run    ")
        print("     [3] : DAQ mode  0: pedestal trigger     ")
        print("                     1: Two hits coincidence for negative signal")
        print("                     2: Two hits coincidence for positive signal")
        print("                     3: Channel B trigger + signal gen")
        print("     [4] : Output file name                  ")
        print("     [5] : Threshold in mV                   ")
        exit()
    print('Number of events per each sub run:',args[1])
    print('Number of subruns for each run   :',args[2])
    print('DAQ mode                         :',args[3])
    print("                     0: pedestal trigger     ")
    print("                     1: Two hits coincidence for negative signal")
    print("                     2: Two hits coincidence for positive signal")
    print("                     3: Channel B trigger + signal gen")
    print('Output file name w/o ".XXX"      :',args[4])
    print('Threshold in mV                  :',args[5])

    num_events=int(args[1])
    num_subruns=int(args[2])
    if num_subruns<1 or num_events<1:
        print("You're wrong!")
        exit()
    daqMode=int(args[3])
    if daqMode<0 or daqMode>3:
        print("Invalid DAQ mode ", daqMode)
        print("                     0: pedestal trigger     ")
        print("                     1: Two hits coincidence for negative signal")
        print("                     2: Two hits coincidence for positive signal")
        print("                     3: Channel B trigger + signal gen")
        exit()

    fname=args[4]
    threshold=float(args[5]) # mV
    if(threshold<0):
        threshold=-1*threshold
    main()
