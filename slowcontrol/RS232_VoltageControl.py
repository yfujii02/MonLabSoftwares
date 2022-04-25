# -*- coding: utf-8 -*-
"""
Created on Mon 29/06/2020
Last Updated: Thurs 09/02/22
Author: Sam Dekkers

Script for ramping up voltage on Keithley 6487 Picoammeter via an RS232 connection

Arguments
[1] = Reset Connection
[2] = Voltage Level Required
[3] = Voltage Increment Before Threshold
[4] = Threshold Voltage
[5] = Voltage Increment After Threshold
[6] = Time Period Required (set to 0 to ignore this) 

"""
import sys
print(sys.path)
import pyvisa as visa
#import visa
import decorator
import time
import numpy as np
import matplotlib.pyplot as plt

args = sys.argv
numberargs = 7
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
        print('Use this file to control voltage source: ')
        print(' ')
        print('sudo python RS232_Comm.py [1] [2] [3] [4] [5] [6]')
        print(' ')
        print(' [1] Reset Connection = 1, else use 0')
        print(' [2] Voltage Level Required (.1f)')
        print(' [3] Voltage Increment Before Threshold Voltage (.1f)')
        print(' [4] Threshold Voltage (above which voltage ramp is in steps of [5]) (int for now)')
        print(' [5] Voltage Increment After Threshold Voltage (.1f)')
        print(' [6] Time Period required (use 0 for indefinite/until next value input) (int?)')
        print(' ')
        sys.exit()
if(ExitFlag==1): sys.exit()
rm = visa.ResourceManager()
print(rm.list_resources())
#Create a connection with the RS232 port listed when printing list_resources
time.sleep(0.5)
instrument=rm.open_resource(u'ASRL/dev/ttyS0::INSTR')
ResetFlag = int(args[1])
instrument.baud_rate=9600
instrument.write_termination = '\n'
instrument.read_termination = '\n'
instrument.timeout=5000
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

TimePeriod = int(args[6])#How long voltage should remain at the specified level (s)

NormIncrement = float(args[3])#Voltage increment below the threshold

ThresholdVoltage = float(args[4])#Voltage level after which it will increment in ThreshIncrement volts

ThreshIncrement = float(args[5]) #Voltage increment after the threshold is reached
###########################SETUP CONNECTION##################################################################
#For when connection issues are occuring, debugging
# if(ResetFlag==1): instrument.write("*RST")

###################################Safely Ramp Down (if not 0 V)############################################
instrument.write("INIT")
instrument.write("FORM:ELEM VSO")
instrument.write("READ?")
time.sleep(0.5)
VoltageRead = str(instrument.read()).split(',')
print(VoltageRead)
VoltageRead = (np.array(VoltageRead)).astype(np.float)[0] #Convert results to float array
print(VoltageRead)
########## Turn off voltage if it's non-zero at the beginning...
if(VoltageRead>0.0):
    while(VoltageRead>0.0):
        voltagecommand='SOUR:VOLT '
        if (VoltageRead>ThresholdVoltage):
            VoltageRead-=0.2
        else:
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
        VoltageRead = np.round(VoltageRead,2)
##########################################################################################################
instrument.write("*RST") #Reset instrument settings on instrument now that voltage safely at 0 V
######################################Start Voltage Ramp###############################################
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
############################################################################################################
if(VoltageCheck==0):
    #instrument.write("*RST") #Reset instrument settings on instrument now that voltage safely at 0 V
    instrument.write("SOUR:VOLT:ILIM 2.5e-4") #Limit current?
    instrument.write("SOUR:VOLT:STAT ON") #Turn voltage source on
    EndFlag=0
    CompFlag=0 #Error flag
    
    #Track both reading and what should be
    #CurrentVoltage = what the voltage should be
    #VoltageReadOut = what the voltage read from the instrument is
    
    while(EndFlag==0):

        CurrentVoltage = VoltageRead #Keep track of voltage as increase it
        if(CurrentVoltage<VoltageLevel):
            while(CurrentVoltage<VoltageLevel):
                voltagecommand='SOUR:VOLT '
                if(CurrentVoltage<ThresholdVoltage):
                    CurrentVoltage+=NormIncrement
                elif(CurrentVoltage>=ThresholdVoltage):
                    CurrentVoltage+=ThreshIncrement

                if(CurrentVoltage>VoltageLevel):
                    CurrentVoltage=VoltageLevel
                CurrentVoltage = np.round(CurrentVoltage,2)
                voltagestr=str(CurrentVoltage)
                voltagecommand+=voltagestr
                instrument.write(voltagecommand)

                #Read whether it actually is this voltage or some other error has occured...
                instrument.write("INIT")
                instrument.write("FORM:ELEM VSO")
                instrument.write("READ?")
                VoltageReadOut = str(instrument.read()).split(',')
                VoltageReadOut = ((np.array(VoltageReadOut)).astype(np.float)[0]) #Convert results to float array
                VoltageReadOut = np.round(VoltageReadOut,2)
                
                #Displays what voltage read from instrument is after sending most recent voltage increment
                sys.stdout.write("\r Voltage: %.2f V" % VoltageReadOut)    
                sys.stdout.flush()
                
                #If the voltage desired doesn't match the voltage read (e.g. instrument sends an error code back) this will
                #set the voltage level target to 0.0 essentially breaking out immediately to ramp down   
                if(VoltageReadOut!=CurrentVoltage and CompFlag==0):
                    print("Error!")
                    if(VoltageReadOut==-999.0): print("Current COMPL Error!") #add more specific codes here if we come across them?
                    VoltageLevel=0.0
                    CompFlag=1

                time.sleep(2)

        elif(CurrentVoltage>VoltageLevel):
            while(CurrentVoltage>VoltageLevel):
                voltagecommand='SOUR:VOLT '
                if(CurrentVoltage>ThresholdVoltage):
                    CurrentVoltage-=ThreshIncrement
                elif(CurrentVoltage<=ThresholdVoltage):
                    voltagecommand='SOUR:VOLT '
                    CurrentVoltage-=NormIncrement

                if(CurrentVoltage<VoltageLevel):
                   CurrentVoltage=VoltageLevel
                CurrentVoltage = np.round(CurrentVoltage,2)
                voltagestr=str(CurrentVoltage)
                voltagecommand+=voltagestr
               
                instrument.write(voltagecommand)
                
                instrument.write("INIT")
                instrument.write("FORM:ELEM VSO")
                instrument.write("READ?")
                VoltageReadOut = str(instrument.read()).split(',')
                VoltageReadOut = (np.array(VoltageReadOut)).astype(np.float)[0] #Convert results to float array
                VoltageReadOut = np.round(VoltageReadOut,2)
                
                sys.stdout.write("\r Voltage: %.2f V" % CurrentVoltage)
                sys.stdout.flush()
                
                if(VoltageReadOut!=CurrentVoltage and CompFlag==0):
                    print("Error!")
                    VoltageLevel=0.0
                    if(VoltageReadOut==-999.0): print("Current COMPL Error!")
                    CompFlag=1
                
                time.sleep(2)

        print("")
        print("Voltage now set to %f V"%(CurrentVoltage))

        if(TimePeriod!=0):
            print("This voltage will be maintained for %d s"%(TimePeriod))
            time.sleep(TimePeriod)
            VoltageLevel=0
        elif(CompFlag==0):
            print("Enter next voltage (0 for ramp down): ")
            VoltageLevel=float(input())
            print("Now setting to %f V"%(VoltageLevel))
        
        if(VoltageLevel==0):
            EndFlag=1
            print("Ramping voltage down now!")

    ###################################Safely Ramp Down (if not 0 V)############################################
    instrument.write("INIT")
    instrument.write("FORM:ELEM VSO")
    instrument.write("READ?")
    VoltageRead = str(instrument.read()).split(',')
    VoltageRead = (np.array(VoltageRead)).astype(np.float)[0] #Convert results to float array
    VoltageRead = np.round(VoltageRead,2)
    if(CompFlag==1): VoltageRead=CurrentVoltage #if an error code is being read, cant use the readout to ramp down so use what it should have been to start ramping down
   
    if(VoltageRead>0.0):
        while(VoltageRead>0.0):
            voltagecommand='SOUR:VOLT '
            if (VoltageRead>ThresholdVoltage):
                VoltageRead-=0.2
            else:
                VoltageRead-=2.0
            if(VoltageRead<2.0):
                VoltageRead=0.0
            voltagestr=str(VoltageRead)
            voltagecommand+=voltagestr
            sys.stdout.write("\r Voltage: %.2f V" %VoltageRead)
            sys.stdout.flush()
            instrument.write(voltagecommand)
            time.sleep(1)
            if(CompFlag==0):
                instrument.write("INIT")
                instrument.write("FORM:ELEM VSO")
                instrument.write("READ?")
                VoltageRead = str(instrument.read()).split(',')
                VoltageRead = (np.array(VoltageRead)).astype(np.float)[0] #Convert results to float array
                VoltageRead = np.round(VoltageRead,2)

    instrument.write("*RST")
    print("")
    print("Voltage ramp complete!")
else:
    print("Voltage level too high!!! (Over 500 V)")
##########################################################################################################
instrument.clear()
