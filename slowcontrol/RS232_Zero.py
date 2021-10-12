# -*- coding: utf-8 -*-
"""
Created on Wed 22/09/21
Last Updated: Wed 22/09/21
Author: Sam Dekkers

Run to safely zero Keithley 6487 Picoammeter via RS232 connection if there
has been some kind of error leaving it at a high voltage
"""
import sys
print(sys.path)
import pyvisa as visa
#import visa
import decorator
import time
import numpy as np
import matplotlib.pyplot as plt
##############################Connection Setup#############################################################
#Set threshold voltage (ramp increment = 0.2 V above, 2.0 V below) 
ThresholdVoltage=37.0 #currently set on assumption of 1 MPPC channel connected

rm = visa.ResourceManager()
print(rm.list_resources())
#Create a connection with the RS232 port listed when printing list_resources
time.sleep(0.5)
instrument=rm.open_resource(u'ASRL/dev/ttyUSB0::INSTR')
ResetFlag = 0
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

###################################Safely Ramp Down (if not 0 V)############################################
instrument.write("INIT")
instrument.write("FORM:ELEM VSO")
instrument.write("READ?")
time.sleep(0.5)
VoltageRead = str(instrument.read()).split(',')
VoltageRead = (np.array(VoltageRead)).astype(np.float)[0] #Convert results to float array

#Ramp down voltage
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

##########################################################################################################
#Reset and clear ready for next connection
instrument.write("*RST") #Reset instrument settings on instrument now that voltage safely at 0 V
instrument.clear()
