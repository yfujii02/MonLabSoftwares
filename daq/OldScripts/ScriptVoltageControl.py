# -*- coding: utf-8 -*-
"""
Created on Mon 29/06/2020
Last Updated: Thurs 23/07/20
Author: Sam Dekkers
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
numberargs = 2
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
        print('Use this file to control voltage source within script: ')
        print(' ')
        print('sudo python RS232_Comm.py [1]')
        print(' ')
        print(' [1] Voltage Level required to maintain')
        print(' Be careful as this won\' reset voltage to 0 V afterwards')
        print(' ')
        sys.exit()

if(ExitFlag==1): sys.exit()

rm = visa.ResourceManager()
print(rm.list_resources())
#Create a connection with the RS232 port listed when printing list_resources
time.sleep(0.5)
instrument=rm.open_resource(u'ASRL/dev/ttyUSB0::INSTR')
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
instrument.write('*IDN?')
print(instrument.read('\n'))

##########################################################################################################
##########################VOLTAGE CONDITIONS#################################################################

VoltageLevel = float(args[1])#83.8 #83.8 #How high voltage is to be set (V)
TimePeriod = 0 #How long voltage should remain at the specified level (s), 0 for indefinite
NormIncrement = 2.0 #voltage increment below the threshold
ThresholdVoltage = 75  #Voltage level after which it will increment in ThreshIncrement volts - set to be integer voltage
ThreshIncrement = 0.1
###########################SETUP CONNECTION##################################################################
###################################Safely Ramp Down (if not 0 V)############################################
instrument.write("INIT")
instrument.write("FORM:ELEM VSO")
instrument.write("READ?")
time.sleep(0.5)
VoltageRead = str(instrument.read()).split(',')
VoltageRead = (np.array(VoltageRead)).astype(np.float)[0] #Convert results to float array
######################################Start Voltage Ramp###############################################
###################################Set Voltage Range######################################################
VoltageCheck=0
if(VoltageLevel==0):
    instrument.write("SOUR:VOLT:RANG 500")
    print("Voltage Range: 500 V")
elif(VoltageLevel<10.0):
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
    instrument.write("SOUR:VOLT:ILIM 2.5e-4") #Limit current?
    instrument.write("SOUR:VOLT:STAT ON") #Turn voltage source on
    EndFlag=0
    while(EndFlag==0):
        CurrentVoltage = VoltageRead #Keep track of voltage as increase it
        if(CurrentVoltage<VoltageLevel):
            while(CurrentVoltage<VoltageLevel):
                if(CurrentVoltage<ThresholdVoltage):
                    voltagecommand='SOUR:VOLT '
                    CurrentVoltage+=NormIncrement

                    if(CurrentVoltage>VoltageLevel):
                        CurrentVoltage=VoltageLevel
                    CurrentVoltage = np.round(CurrentVoltage,2)
                    voltagestr=str(CurrentVoltage)
                    voltagecommand+=voltagestr
                    sys.stdout.write("\r Voltage: %.2f V" % CurrentVoltage)
                    sys.stdout.flush()
                    #print(voltagecommand)
                    instrument.write(voltagecommand)
                    time.sleep(2)

                elif(CurrentVoltage>=ThresholdVoltage):
                    voltagecommand='SOUR:VOLT '
                    CurrentVoltage+=ThreshIncrement

                    if(CurrentVoltage>VoltageLevel):
                        CurrentVoltage=VoltageLevel

                    CurrentVoltage = np.round(CurrentVoltage,2)
                    voltagestr=str(CurrentVoltage)
                    voltagecommand+=voltagestr
                    sys.stdout.write("\r Voltage: %.2f V" % CurrentVoltage)
                    sys.stdout.flush()
                    #print(voltagecommand)
                    instrument.write(voltagecommand)
                    time.sleep(2)

        elif(CurrentVoltage>VoltageLevel):
            while(CurrentVoltage>VoltageLevel):
                if(CurrentVoltage>ThresholdVoltage):
                    voltagecommand='SOUR:VOLT '
                    CurrentVoltage-=ThreshIncrement

                    if(CurrentVoltage<VoltageLevel):
                       CurrentVoltage=VoltageLevel
                    CurrentVoltage = np.round(CurrentVoltage,2)
                    voltagestr=str(CurrentVoltage)
                    voltagecommand+=voltagestr
                    sys.stdout.write("\r Voltage: %.2f V" % CurrentVoltage)
                    sys.stdout.flush()
                    #print(voltagecommand)
                    instrument.write(voltagecommand)
                    time.sleep(2)

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
                    #print(voltagecommand)
                    instrument.write(voltagecommand)
                    time.sleep(2)

        VoltageRead=VoltageLevel
        print("")
        print("Voltage now set to %f V"%(CurrentVoltage))
        if(VoltageLevel==0): instrument.write("*RST")
        EndFlag=1
