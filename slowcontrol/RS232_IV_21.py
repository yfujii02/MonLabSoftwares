# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:15:30 2021

@author: samde
"""


import sys
print(sys.path)
import pyvisa as visa
#import visa
import decorator
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

args = sys.argv
numberargs = 9
ExitFlag=0
if(len(args)!=numberargs): 
    if(len(args)>1):
        if(args[1]!='-help'):
            print("Wrong Input - use -help as argument [1] for full breakdown")
            sys.exit()
    else:print("Wrong Input - use -help as argument [1] for full breakdown") 
    ExitFlag=1
if(len(args)>1):
    if (args[1]=='-help'):
        print('Use this file to control voltage source while taking IV curve measurements: ')
        print(' ')
        print('python3.7 RS232_IV_21.py [1] [2] [3] [4] [5] [6] [7] [8]')
        print(' ')
        print(' [1] Reset Connection = 1, else use 0')
        print(' [2] Voltage Level Required (volts .1f)')
        print(' [3] Voltage Increment Before Threshold Voltage (volts .1f)')
        print(' [4] Threshold Voltage (above which voltage ramp is in steps of [5]) (volts .1f)')
        print(' [5] Voltage Increment After Threshold Voltage (volts .1f)')
        print(' [6] File name identifier string e.g. "Current1" for Current1.txt measurement output')
        print(' [7] No. current measurements at each voltage (int)')
        print(' [8] Trigger delay between measurements (seconds .1f)')
        print(' ')
        sys.exit()
if(ExitFlag==1): sys.exit()

dOffset=256
############################################Initialisation Commands##############################################
rm = visa.ResourceManager()
print(rm.list_resources())
#Create a connection with the RS232 port listed when printing list_resources
time.sleep(0.5)
instrument=rm.open_resource(u'ASRL/dev/ttyUSB0::INSTR')
ResetFlag = int(args[1])
instrument.baud_rate=9600
instrument.write_termination = '\n'
instrument.read_termination = '\n'
measPoints=int(args[7])+dOffset #no. current measurements at each voltage
triggerDelay = float(args[8])
instrument.timeout=100*measPoints+triggerDelay
#Setup command character terminations - basically the settings so that the PC and instrument
#recognise the end of a command - this is instrument dependent unfortunately and not always
#clear as to what it should be set to.

#Send a query command for instrument identification to test both read and write commands are
#functioning at both ends

#Display settings
print("%%%%%%%%%RS232-USB Configuration Settings%%%%%%%%%%%")
# print("IP Address: 192.168.10.200")
# print("Connection Port: 1234")
print("Write Termination Character: \\n")
print("Read Termination Character: \\n")
print("Instrument Address: 22")

if(ResetFlag==1):
    print(instrument.write("*RST"))
    print("RESET")

instrument.write('*IDN?')
print(instrument.read('\n'))

##########################################################################################################
##########################VOLTAGE CONDITIONS#################################################################

VoltageLevel = float(args[2])#How high voltage is to be set (V)

FileString =  str(args[6])
FileName = "/home/comet/work/pico/IVCurveData21/"+str(args[6])+".txt" #file name for current measurements
FileNameConc ="/home/comet/work/pico/IVCurveData21/"+str(args[6]) #file name for current measurements without .txt
NormIncrement = float(args[3])# 1.0 #voltage increment below the threshold

ThresholdVoltage = float(args[4]) #Voltage level after which it will increment in ThreshIncrement volts

ThreshIncrement = float(args[5]) #Voltage increment above the threshold

Increments = ThresholdVoltage+int((VoltageLevel-ThresholdVoltage)/ThreshIncrement)
###########################SETUP CONNECTION##################################################################
#For when connection issues are occuring
#if(ResetFlag==1): instrument.write("*RST")
#When connection issues occuring, turn off and run again with reset flag on (not sure if this extra one here is necessary)
###################################Safely Ramp Down (if not 0 V)############################################
instrument.write("INIT")
instrument.write("FORM:ELEM VSO")
instrument.write("READ?")
time.sleep(0.5)
VoltageRead = str(instrument.read()).split(',')
VoltageRead = (np.array(VoltageRead)).astype(np.float)[0] #Convert results to float array

if(VoltageRead>0.0):
    while(VoltageRead>ThresholdVoltage):
      voltagecommand='SOUR:VOLT '
      VoltageRead-=0.2
      voltagestr=str(VoltageRead)
      voltagecommand+=voltagestr
      sys.stdout.write("\r Voltage: %.2f V" % VoltageRead)
      sys.stdout.flush()
      instrument.write(voltagecommand)
      time.sleep(1)
      instrument.write("INIT")
      instrument.write("FORM:ELEM VSO")
      instrument.write("READ?")
      VoltageRead = str(instrument.read()).split(',')
      VoltageRead = (np.array(VoltageRead)).astype(np.float)[0] #Convert results to float array
      
    while(VoltageRead>0.0):
        voltagecommand='SOUR:VOLT '
        VoltageRead-=2.0
        if(VoltageRead<2.0):
           VoltageRead=0.0
        voltagestr=str(VoltageRead)
        voltagecommand+=voltagestr
        sys.stdout.write("\r Voltage: %.2f V" % VoltageRead)
        sys.stdout.flush()
        instrument.write(voltagecommand)
        time.sleep(1)
        instrument.write("INIT")
        instrument.write("FORM:ELEM VSO")
        instrument.write("READ?")
        VoltageRead = str(instrument.read()).split(',')
        VoltageRead = (np.array(VoltageRead)).astype(np.float)[0] #Convert results to float array
        
##########################################################################################################
instrument.write("*RST") #Reset instrument settings on instrument now that voltage safely at 0 V
############################################Measurement Function####################################
#Define data arrays
sourceVoltage=[]
averageCurrent=[]
minCurrent=[]
maxCurrent=[]
stdCurrent=[]
FitCurrent=[]

CurrentAllData=np.array([])
TimeAllData=[]
amplitude=0
MaxI=0


def TakeMeasurement(voltage):
    global amplitude
    global MaxI
    sourceVoltage.append(voltage)
    print("Source V = ",voltage)
    time.sleep(2+triggerDelay)
    instrument.write("SYST:ZCH OFF") #Turn zero checking off (zero checking is on it seems normally for changing the circuit)
    instrument.write("SYST:AZER:STAT OFF") #
    instrument.write("DISP:ENAB OFF") #Turn display off while setting up buffer
    instrument.write("*CLS")
    instrument.write("TRAC:POIN "+str(measPoints)) #buffer expects 'n=measPoints' measurements
    instrument.write("TRAC:CLE") #Clear buffer
    instrument.write("TRAC:FEED:CONT NEXT") #I think this is for feeding measurements in one after the other
    instrument.write("STAT:MEAS:ENAB 512")
    instrument.write("*SRE 1")
    instrument.write("*OPC?") #Check if ready
    instrument.read() #Important to read this check otherwise this will be read into measurement array and the measurements will be read in the next iteration
    instrument.write("INIT") #Start measurements
    instrument.write("DISP:ENAB ON") #Turn display back on
    instrument.write("TRAC:DATA?") #Request all stored readings
    results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
    #print(results)
    RArray = (np.array(results)).astype(np.float) #Convert results to float array
    #print(RArray)

    CArray = RArray[0::3]#Extract current measurements
    TArray = RArray[1::3]#Extract buffer timestamps
    VArray = RArray[2::3]#Read voltage source data (not needed so might take this out to improve read speeds)
    #print(CArray)
    #print(TArray)
    #print(VArray)

    MeanCurrent = np.mean(CArray)#Calcuate mean current
    averageCurrent.append(MeanCurrent)#Store mean current in array
    minCurrent.append(np.min(CArray)) #Store min
    maxCurrent.append(np.max(CArray)) #Store max
    stdCurrent.append(np.std(CArray))

    #Global parameter updates
    amplitude = (np.max(CArray[dOffset:])-np.min(CArray[dOffset:]))/2.0
    MaxI=np.max(CArray[dOffset:])
   
    Data = np.array([sourceVoltage,averageCurrent,minCurrent,maxCurrent,stdCurrent])
    Data=np.transpose(Data)
    
    #tempCArray=np.array(CArray)
    #tempCArray.insert(0,voltage)
    #np.append(CurrentAllData,tempCArray)
    TimeAllData.append(TArray)

    np.savetxt(FileName, Data, fmt='%.18e', delimiter=' ', newline='\n', header='IV Curve Data: SourceVoltage,AveCurrent,Min,Max,Std', footer='', comments='# ')
    #np.savetxt(FileNameConc+"_CurrentData.txt",np.array(CurrentAllData), fmt='%.18e', delimiter=' ', newline='\n', header='CurrentWaveforms', footer='', comments='# ')
    #np.savetxt(FileNameConc+"_TimeData.txt",TArray, fmt='%.18e', delimiter=' ', newline='\n', header='TimeStamps', footer='', comments='# ')
    return Data, CArray, TArray

def InitFitFunc(x,c,d):
    return amplitude*np.sin(50*x*2*np.pi+c)+d

def FitFunc(x,a,b,c,d):
    return a*np.sin(b*x*2*np.pi+c)+d

def UpdatePlot(data,carray,tarray):
    #Update plot (time vs current) using DataArray from measurement function
    srcV=sourceVoltage
    avC=averageCurrent
    stdC=stdCurrent
    
    #Frequency guess in data based on fourier transform
    ff = np.fft.fftfreq(len(tarray),(tarray[1]-tarray[0]))
    Fcc = abs(np.fft.fft(carray))
    FreqGuess = abs(ff[np.argmax(Fcc[1:])+1])
    #Amplitude guess based on half of difference between max and min current data
    AmpGuess = amplitude
    #Offset guess based on mean of current data
    OffsetGuess = np.mean(carray)

    params, params_covariance = optimize.curve_fit(InitFitFunc, tarray[dOffset:], carray[dOffset:],sigma=5e-10*np.ones(len(carray[dOffset:])), p0=[0.,OffsetGuess])
    
    #optimize_func = lambda x: FitFunc(tarray,AmpGuess,FreqGuess,params[0],OffsetGuess) - carray
    
    GuessParam = np.array([AmpGuess,FreqGuess,params[0],OffsetGuess])
    print("Guess Params")
    print(GuessParam)
    
    #est_amp,est_freq,est_phase,est_offset=optimize.leastsq(optimize_func, GuessParam)[0]
    #data_fit = FitFunc(tarray,est_amp,est_freq,est_phase,est_offset)
    print("FitParams")
    #print(params)
    #params_fit=np.array([est_amp,est_freq,est_phase,est_offset])
    #print(params_fit) 
    FitCurrent.append(params[1])#Using the initial fit as it is better for now...
    #FitCurrent.append(est_offset)
    
    #ax1.plot(srcV,avC,color='k',alpha=0.1)
    #plt.errorbar(srcV,avC,stdC,color='k')
    #ax1.scatter(srcV,avC,color='r')
    
    ax1.scatter(srcV,FitCurrent,color='b')
    ax1.set_xlabel("Voltage (V)")
    ax1.set_ylabel("Current (A)")
    ax1.set_title("IV Curve")
    
    #Plot fit
    ax2.clear()
    ax2.scatter(tarray,carray)
    ax2.plot(tarray[dOffset:],InitFitFunc(tarray[dOffset:],params[0],params[1]),color='r')
    #ax2.plot(tarray,data_fit,color='b')
    ax2.set_title("I Data at %s V" %(voltagestr))
    ax2.set_xlabel("Time (s)")
    plt.draw()
    plt.pause(0.0001)
    FitData=np.array([srcV,FitCurrent])
    FitData=np.transpose(FitData)
    np.savetxt(FileNameConc+'_FitData.txt', FitData, fmt='%.18e', delimiter=' ', newline='\n', header='IV Curve Data: SourceVoltage,FitCurrent', footer='', comments='# ')

######################################Start Voltage Ramp###############################################
#Setup
#ZeroMeasurement()
instrument.write("FORM:ELEM READ,TIME,VSO")#Read current, timestamp, voltage
instrument.write("TRIG:DEL 0.0") #Trigger delay of 0;;; IT'S NOT WORKING!!
instrument.write("TRIG:COUN "+str(measPoints)) #Trigger count of 25
instrument.write("NPLC .01")
instrument.write("RANG 2e-6") #Current range of 2 uA
instrument.write("AVER:COUN 100") #100 point averaged filter
instrument.write("AVER:TCON REP") #Repeating filter
instrument.write("AVER:ON") #Turn filter on

###################################Set Voltage Range######################################################
VoltageCheck=0
if(VoltageLevel<10.0):
    instrument.write("SOUR:VOLT:RANG 10") #Set voltage range to 10 V range
    print("Voltage Range: 10 V")
elif(VoltageLevel>=10.0 and VoltageLevel<50.0):
    instrument.write("SOUR:VOLT:RANG 50") #Set voltage range to 50 V range
    print("Voltage Range: 50 V")
elif(VoltageLevel>=50.0 and VoltageLevel<500.0):
    instrument.write("SOUR:VOLT:RANG 500") #Set voltage range to 500 V range
    print("Voltage Range: 500 V")
else:
    VoltageCheck=1
instrument.write("SOUR:VOLT:ILIM 2.5e-4") #Limit current?
instrument.write("SOUR:VOLT:STAT ON") #Turn voltage source on
############################################################################################################
EndFlag=0
fig,(ax1,ax2) = plt.subplots(1,2)
while(EndFlag==0):
    if(VoltageCheck==0):
        #instrument.write("*RST") #Reset instrument settings on instrument now that voltage safely at 0 V
        #instrument.write("SOUR:VOLT:ILIM 2.5e-4") #Limit current?
        #instrument.write("SOUR:VOLT:STAT ON") #Turn voltage source on
        CurrentVoltage = VoltageRead #Keep track of voltage as increase it
        if(CurrentVoltage<VoltageLevel):
            while(CurrentVoltage<VoltageLevel):
                voltagecommand='SOUR:VOLT '
                if(CurrentVoltage<1.0):
                    CurrentVoltage+=0.2
                elif(CurrentVoltage<ThresholdVoltage and CurrentVoltage+NormIncrement>ThresholdVoltage ):
                    CurrentVoltage=ThresholdVoltage
                elif(CurrentVoltage<ThresholdVoltage):
                    CurrentVoltage+=NormIncrement
                elif(CurrentVoltage>=ThresholdVoltage):
                    CurrentVoltage+=ThreshIncrement
    
                if(CurrentVoltage>VoltageLevel):
                    CurrentVoltage=VoltageLevel
                CurrentVoltage = np.round(CurrentVoltage,2)
                voltagestr=str(CurrentVoltage)
                voltagecommand+=voltagestr
                sys.stdout.write("\r Voltage: %.2f V" % CurrentVoltage)
                sys.stdout.flush()
                instrument.write(voltagecommand) ##### voltage set
                #time.sleep(5)
                CurrentDataMeasurement,CurrentCurrent,CurrentTime = TakeMeasurement(CurrentVoltage)
                print(CurrentDataMeasurement[len(CurrentDataMeasurement)-1])
                UpdatePlot(CurrentDataMeasurement,CurrentCurrent,CurrentTime)
                time.sleep(1)
                if (MaxI>0.3*2e-6):
                    instrument.write("RANG 2e-5") #Current range of 20 uA
                    print("Range changed to 2e-5")
                if (MaxI>0.3*2e-5):
                    instrument.write("RANG 2e-4") #Current range of 200 uA
                    print("Range changed to 2e-4")

        elif(CurrentVoltage>VoltageLevel):
                while(CurrentVoltage>VoltageLevel):
                    if(CurrentVoltage>ThresholdVoltage):
                        voltagecommand='SOUR:VOLT '
                        CurrentVoltage-=ThreshIncrement
                    elif(CurrentVoltage<=ThresholdVoltage):
                        voltagecommand='SOUR:VOLT '
                        CurrentVoltage-=NormIncrement
    
                    if(CurrentVoltage<VoltageLevel):
                       CurrentVoltage=VoltageLevel
                    CurrentVoltage = np.round(CurrentVoltage,2)
                    voltagestr=str(CurrentVoltage)
                    voltagecommand+=voltagestr
                    sys.stdout.write("\r Voltage: %.2f V" % CurrentVoltage)
                    sys.stdout.flush()
                    instrument.write(voltagecommand)
                    time.sleep(1)
    
    
        VoltageRead=VoltageLevel
        print("")
        print("Voltage now set to %f V"%(CurrentVoltage))
    
        print("Enter next voltage (0 for ramp down): ")
        VoltageLevel=float(input())
        print("Now setting to %f V"%(VoltageLevel))
    
        if(VoltageLevel==0):
          EndFlag=1
          print("Ramping voltage down now!")
    else: 
        EndFlag=1
        
fig.savefig('/home/comet/work/pico/IVCurveData21/'+FileString+'.png')
###################################Safely Ramp Down (if not 0 V)############################################
instrument.write("TRIG:COUN 1") #Trigger count of 1
instrument.write("TRAC:POIN 1") #Buffer expects 1 measurement
instrument.write("INIT")
instrument.write("TRAC:CLE") #Clear buffer
instrument.write("FORM:ELEM READ, VSO")
instrument.write("READ?")
time.sleep(0.5)

while(VoltageRead>ThresholdVoltage):
  voltagecommand='SOUR:VOLT '
  VoltageRead-=0.2
  voltagestr=str(VoltageRead)
  voltagecommand+=voltagestr
  sys.stdout.write("\r Voltage: %.2f V" % VoltageRead)
  sys.stdout.flush()
  instrument.write(voltagecommand)
  time.sleep(1)

while(VoltageRead>0.0):
    voltagecommand='SOUR:VOLT '
    VoltageRead-=2.0
    if(VoltageRead<=1.9):
       VoltageRead=0.0
    voltagestr=str(VoltageRead)
    voltagecommand+=voltagestr
    sys.stdout.write("\r Voltage: %.2f V" % VoltageRead)
    sys.stdout.flush()
    instrument.write(voltagecommand)
    time.sleep(1)

##########################################################################################################
instrument.write("*RST") #Reset instrument settings on instrument now that voltage safely at 0 V
##########################################################################################################
instrument.clear()
