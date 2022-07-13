
import sys
print(sys.path)
import pyvisa as visa
import decorator
import time
import numpy as np
import matplotlib.pyplot as plt

RangeVoltage=0.0
TargetVoltage = 0.0
ThresholdVoltage = 37.0
ThreshIncrement = 0.2
NormIncrement = 2.0
ResetFlag = 0
CompFlag = 0
EndFlag = 1

def InitParameters(Target,Thresh,ThreshInc,NormInc,Reset,Win):
    global TargetVoltage
    global ThresholdVoltage
    global ThreshIncrement
    global NormIncrement
    global instrument
 
    rm = visa.ResourceManager() 
    print(rm.list_resources()) #shows available connections (good to see if wrong USB port is being used - pick whichever the RS232 is being listened to on)
    #If no ports are listed, might require a reset of the picoammeter
    time.sleep(0.5)
    if(Win==0): instrument=rm.open_resource(u'ASRL/dev/ttyUSB0::INSTR')
    elif(Win==1): instrument=rm.open_resource(u'ASRL19::INSTR') #for use with windows pc

    ResetFlag = int(Reset) #reset flag helps for rectifying some communication issues - don't use if voltage already high in case you damage whatever is connected

    #Instrument communication parameters

    instrument.baud_rate=9600 #how many bits per second are coommunicated - 9600 is used for RS232 communication on the 6487

    #Setup command character terminations - basically the settings so that the PC and instrument
    #recognise the end of a command - this is instrument dependent unfortunately and not always
    #clear as to what it should be set to.
    instrument.write_termination = '\n'
    instrument.read_termination = '\n'

    instrument.timeout=5000 #timeout occurs if nothing is heard back in this many ms

    #First need to send a query command for instrument identification to test both read and write commands are
    #functioning at both ends - if everything is okay then instrument.write('*IDN?') will return the instrument
    #details when we print instrument.read('\n')

    #Display settings
    print("%%%%%%%%%RS232-USB Configuration Settings%%%%%%%%%%%")
    print("Write Termination Character: \\n")
    print("Read Termination Character: \\n")
    print("Instrument Address: 22")

    #Reset before starting if we want - not really necessary if everything is working well
    if(ResetFlag==1):
        print(instrument.write("*RST"))
        print("RESET")

    instrument.write('*IDN?')
    print(instrument.read('\n'))

    TargetVoltage = Target
    ThresholdVoltage = Thresh
    ThreshIncrement = ThreshInc
    NormIncrement = NormInc

def UpdateParameters(T,TInc,Inc):
    global ThreshIncrement
    global ThresholdVoltage
    global NormIncrement
    ThreshIncrement = TInc
    ThresholdVoltage = T
    NormIncrement = Inc    
def Clear():
    instrument.clear()
###################################Set Voltage Range Function######################################################
def SetRange(VoltageLevel):
    global RangeVoltage
    RangeVoltage = 0.0
    if(VoltageLevel<10.0):
        instrument.write("SOUR:VOLT:RANG 10") #Set voltage range to 10 V range
        print("Voltage Range: 10 V")
        RangeVoltage = 10.0
    elif(VoltageLevel>=10.0 and VoltageLevel<50.0):
        instrument.write("SOUR:VOLT:RANG 50") #Set voltage range to 50 V range
        print("Voltage Range: 50 V")
        RangeVoltage = 50.0
    elif(VoltageLevel>=50.0 and VoltageLevel<500.0):
        instrument.write("SOUR:VOLT:RANG 500") #Set voltage range to 500 V range
        print("Voltage Range: 500 V")
        RangeVoltage = 500.0
    else:
        print("Error! Voltage must be between 0.0 and 500.0 V!")
        sys.exit()
##################################Read Voltage Function##########################################################################
def ReadVoltage():
    instrument.write("INIT")
    instrument.write("FORM:ELEM VSO")
    instrument.write("READ?")
    VoltageRead = str(instrument.read()).split(',')
    VoltageRead = ((np.array(VoltageRead)).astype(float)[0]) #Convert results to float array
    VoltageRead = np.round(VoltageRead,2)
    return VoltageRead
#################################Ramp Down Function##############################################################################
def ZeroVoltage(VoltageRead):
    while(VoltageRead>0.0):
        if(VoltageRead>ThresholdVoltage):
            VoltageRead-=ThreshIncrement
        else:
            VoltageRead-=NormIncrement
        
        if(VoltageRead<2.0):
            VoltageRead=0.0
        
        voltagecommand='SOUR:VOLT '+str(VoltageRead)
        
        sys.stdout.write("\r Voltage: %.2f V" %VoltageRead)
        sys.stdout.flush()
        
        instrument.write(voltagecommand)
        time.sleep(1)
        
        if(CompFlag==0): VoltageRead = ReadVoltage() #Need to check whether comp error occured or not otherwise can't ramp down with checking read voltage
    instrument.write("*RST") #Reset instrument settings on instrument now that voltage safely at 0 V
