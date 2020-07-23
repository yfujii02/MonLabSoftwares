# -*- coding: utf-8 -*-
"""
Created on Mon June 29

@author: Sam Dekkers
CAUTION: THIS IS FOR A SETUP WITH 2 SiPM IN SERIES - CHECK VOLTAGE REQUIREMENTS

Script for communicating with Keithley 6487 Picoammeter via an RS232-USB cable

Keithley 6847 Picoammeter Manual including SCPI commands:

https://www.tek.com/low-level-sensitive-and-specialty-instruments/series-6400-picoammeters-manual/model-6487-picoammeter\

PyVISA Package Information:

https://pyvisa.readthedocs.io/en/master/introduction/index.html

"""

import pyvisa as visa
import decorator
import time
import numpy as np
import matplotlib.pyplot as plt


###########################SETUP CONNECTION##################################################################
rm = visa.ResourceManager()
print(rm.list_resources())
#Create a connection with the RS232 port listed when printing list_resources
time.sleep(0.5)
instrument=rm.open_resource(u'ASRL/dev/ttyUSB0::INSTR')

#instrument.baud_rate(9600)
instrument.write_termination = '\n'
instrument.read_termination = '\n'
instrument.timeout=20000
instrument.baud_rate=9600
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
instrument.write("*RST") #Reset RS232 settings on instrument
instrument.write('*IDN?')
print(instrument.read())


#Voltage Sweep with 25 current measurements per voltage each coming from 100 averaged measurements
#Voltage range is from 0 to UpperVoltage in increments of 1 V until ThreshVoltage where step size changes to ThreshIncrement

plt.ion() #Create plot to update at each time step

#Data Arrays
averageCurrent=[]#Store average current value at each voltage step
sourceVoltage=[] #Store source voltage at each step
minCurrent=[] #Store minimum current measurement
maxCurrent=[] #Store maxumum current measurement
stdCurrent=[] #store standard deviation of measurements

#Picoammeter Setup Instructions
#Start by checking voltage and safely decreasing to start measurements if it isn't 0 V
instrument.write("SOUR:VOLT:RANG 500") #Set voltage range to 500 V range
instrument.write("INIT")
instrument.write("FORM:ELEM READ, VSO")
instrument.write("READ?")
results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
print(results)
RArray = (np.array(results)).astype(np.float) #Convert results to float array
print(RArray[1])
VoltageRead=RArray[1];
if(VoltageRead>0.0):
    while(VoltageRead>70.0):
      voltagecommand='SOUR:VOLT '
      VoltageRead-=0.2
      voltagestr=str(VoltageRead)
      voltagecommand+=voltagestr
      print(voltagecommand)
      instrument.write(voltagecommand)
      time.sleep(1)
      instrument.write("INIT")
      instrument.write("FORM:ELEM READ, VSO")
      instrument.write("READ?")
      results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
      RArray = (np.array(results)).astype(np.float) #Convert results to float array
      print("Source Voltage Measurement:")
      print(RArray[1])
    while(VoltageRead>0.0):
        voltagecommand='SOUR:VOLT '
        VoltageRead-=2.0
        voltagestr=str(VoltageRead)
        voltagecommand+=voltagestr
        print(voltagecommand)
        instrument.write(voltagecommand)
        time.sleep(1)
        instrument.write("INIT")
        instrument.write("FORM:ELEM READ, VSO")
        instrument.write("READ?")
        results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
        print(results)
        RArray = (np.array(results)).astype(np.float) #Convert results to float array
        print(RArray[1])


instrument.write("*RST") #Reset RS232 settings on instrument

###############################CHANGE#################################
#Set Upper and Change in Increment Voltages
UpperVoltage = 84.8 #Make sure divisible by thresh amount
VoltageIncrement=1.0
ThreshVoltage = 73#Put an integer number of volts for now
ThreshIncrement = 0.2
Increments = int(ThreshVoltage/VoltageIncrement)+int((UpperVoltage-ThreshVoltage)/ThreshIncrement)
print("Steps = %d" %Increments)
print("Estimated completion time = %.2f minutes" %(Increments*5/60))
FileName = 'Problematic_IV_2SiPM_Upper85V_CableBadSiPmNewBaseIsolatingAlFoil.dat'
######################################################################


instrument.write("FORM:ELEM READ,TIME,VSO")#Read current, timestamp and voltage source at each step
instrument.write("TRIG:DEL 0") #Trigger delay of 0
instrument.write("TRIG:COUN 25") #Trigger count of 25
instrument.write("NPLC .01")
instrument.write("RANG 2e-6") #Current range of 2 uA
# instrument.write("MED:RANK 5")
# instrument.write("MED ON")
instrument.write("AVER:COUN 100") #100 point averaged filter
instrument.write("AVER:TCON REP") #Repeating filter
instrument.write("AVER ON") #Turn filter on
instrument.write("SOUR:VOLT:RANG 500") #Set voltage range to 10 V range
voltage=-1.0



for i in range(Increments):
    #Construct source voltage message for each voltage in the sweep
    voltagecommand='SOUR:VOLT '
    if(i<=ThreshVoltage):
      voltage+=VoltageIncrement
    else:
      voltage+=ThreshIncrement
    if(voltage>UpperVoltage):
        break


    voltagestr=str(voltage)
    voltagecommand+=voltagestr
    print(voltagecommand)
    sourceVoltage.append(voltage)
    instrument.write(voltagecommand)

    instrument.write("SOUR:VOLT:ILIM 2.5e-4") #Limit current?
    instrument.write("SOUR:VOLT:STAT ON") #Turn voltage source on
    time.sleep(2)
    instrument.write("SYST:ZCH OFF") #Turn zero checking off (zero checking is on it seems normally for changing the circuit)
    instrument.write("SYST:AZER:STAT OFF") #

    instrument.write("DISP:ENAB OFF") #Turn display off while setting up buffer
    instrument.write("*CLS")

    instrument.write("TRAC:POIN 25") #buffer expects 25 measurements
    instrument.write("TRAC:CLE") #Clear buffer
    instrument.write("TRAC:FEED:CONT NEXT") #I think this is for feeding measurements in one after the other
    instrument.write("STAT:MEAS:ENAB 512")
    instrument.write("*SRE 1")
    instrument.write("*OPC?") #Check if ready
    print(instrument.read())
    instrument.write("INIT") #Start measurements
    instrument.write("DISP:ENAB ON") #Turn display back on
    instrument.write("TRAC:DATA?") #Request all stored readings
    results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
    print(results)
    RArray = (np.array(results)).astype(np.float) #Convert results to float array
    CArray = RArray[0::3]#Extract current measurements
    TArray = RArray[1::3]#Extract buffer timestamps
    VArray = RArray[2::3]#Read voltage source data (not needed so might take this out to improve read speeds)
    print(CArray)
    print(TArray)
    print(VArray)

    MeanCurrent = np.mean(CArray)#Calcuate mean current
    averageCurrent.append(MeanCurrent)#Store mean current in array
    minCurrent.append(np.min(CArray)) #Store min
    maxCurrent.append(np.max(CArray)) #Store max
    stdCurrent.append(np.std(CArray))
    Data = np.array([sourceVoltage,averageCurrent,minCurrent,maxCurrent,stdCurrent])
    Data=np.transpose(Data)
    np.savetxt(FileName, Data, fmt='%.18e', delimiter=' ', newline='\n', header='IV Curve Data: SourceVoltage,AveCurrent,Min,Max,Std', footer='', comments='# ')
    #Update plot (time vs current)
    plt.plot(sourceVoltage,averageCurrent)
    plt.draw()
    plt.title("Current Measurements at %s V" %(voltagestr))
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.pause(0.0001)


Data = np.array([sourceVoltage,averageCurrent,minCurrent,maxCurrent,stdCurrent])
Data=np.transpose(Data)
# np.savetxt('IVData.dat', Data, fmt='%.18e', delimiter=' ', newline='\n', header='IV Curve Data: SourceVoltage,AveCurrent,Min,Max,Std', footer='', comments='# ', encoding=None)
np.savetxt(FileName, Data, fmt='%.18e', delimiter=' ', newline='\n', header='IV Curve Data: SourceVoltage,AveCurrent,Min,Max,Std', footer='', comments='# ')
#Ideally end by checking voltage and safely decreasing to start measurements if it isn't 0 V
#instrument.write("SOUR:VOLT:RANG 50") #Set voltage range to 10 V range
instrument.write("TRIG:COUN 1") #Trigger count of 1
instrument.write("TRAC:POIN 1") #buffer expects 1 measurements
instrument.write("INIT")
instrument.write("FORM:ELEM READ, VSO")
instrument.write("READ?")
results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
print(results)
RArray = (np.array(results)).astype(np.float) #Convert results to float array
print(RArray[1])
VoltageRead=RArray[1];

if(VoltageRead>0.0):
    while(VoltageRead>70.0):
      voltagecommand='SOUR:VOLT '
      VoltageRead-=0.2
      voltagestr=str(VoltageRead)
      voltagecommand+=voltagestr
      print(voltagecommand)
      instrument.write(voltagecommand)
      time.sleep(3)
      instrument.write("INIT")
      instrument.write("FORM:ELEM READ, VSO")
      instrument.write("READ?")
      results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
      RArray = (np.array(results)).astype(np.float) #Convert results to float array
      print("Source Voltage Measurement:")
      print(RArray[1])
    while(VoltageRead>0.0):
        voltagecommand='SOUR:VOLT '
        VoltageRead-=2.0
        if(VoltageRead<2.0):
           VoltageRead=0.0
        voltagestr=str(VoltageRead)
        voltagecommand+=voltagestr
        print(voltagecommand)
        instrument.write(voltagecommand)
        time.sleep(3)
        instrument.write("INIT")
        instrument.write("FORM:ELEM READ, VSO")
        instrument.write("READ?")
        results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
        print(results)
        RArray = (np.array(results)).astype(np.float) #Convert results to float array
        print(RArray[1])
instrument.write("*RST") #Reset RS232 settings on instrument



#Ideally end by checking voltage and safely decreasing to start measurements if it isn't 0 V
#instrument.write("SOUR:VOLT:RANG 500") #Set voltage range to 10 V range
instrument.write("INIT")
instrument.write("FORM:ELEM READ, VSO")
instrument.write("READ?")
instrument.write("++read eoi")
results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
print(results)
RArray = (np.array(results)).astype(np.float) #Convert results to float array
print(RArray[1])
VoltageRead=RArray[1];
if(VoltageRead>0.0):
    while(VoltageRead>36.0):
      voltagecommand='SOUR:VOLT '
      VoltageRead-=0.2
      voltagestr=str(VoltageRead)
      voltagecommand+=voltagestr
      print(voltagecommand)
      instrument.write(voltagecommand)
      time.sleep(1)
      instrument.write("INIT")
      instrument.write("FORM:ELEM READ, VSO")
      instrument.write("READ?")
      instrument.write("++read eoi")
      results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
      RArray = (np.array(results)).astype(np.float) #Convert results to float array
      print("Source Voltage Measurement:")
      print(RArray[1])
    while(VoltageRead>0.0):
        voltagecommand='SOUR:VOLT '
        VoltageRead-=2.0
        voltagestr=str(VoltageRead)
        voltagecommand+=voltagestr
        print(voltagecommand)
        instrument.write(voltagecommand)
        time.sleep(1)
        instrument.write("INIT")
        instrument.write("FORM:ELEM READ, VSO")
        instrument.write("READ?")
        instrument.write("++read eoi")
        results = (str(instrument.read()).replace("A","")).split(',') #Read and format results
        print(results)
        RArray = (np.array(results)).astype(np.float) #Convert results to float array
        print(RArray[1])
instrument.write("*RST") #Reset instrument settings on instrument
raw_input("Press Enter to continue...")
