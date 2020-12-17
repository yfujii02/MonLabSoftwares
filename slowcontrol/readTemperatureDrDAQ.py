#
# Copyright (C) 2020 Pico Technology Ltd. See LICENSE file for terms.
#
# USBDRDAQ SCOPE BLOCK MODE EXAMPLE
# This example opens a UsbDrDaq driver device, sets up the scope channel and a trigger then collects a single block of data.
# This data is then plotted as mV against time in ns.

import ctypes
import time
from picosdk.usbDrDaq import usbDrDaq as drDaq
import numpy as np
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
from datetime import datetime

# Create chandle and status ready for use
status = {}
chandle = ctypes.c_int16()
filename='temperature.txt'

ideal_no_of_samples = 1024
interval = 10 # default time interval (sec)
#interval = 2 # default time interval (sec)

# Opens the device
def init():
    global status
    global chandle
    status["openunit"] = drDaq.UsbDrDaqOpenUnit(ctypes.byref(chandle))
    assert_pico_ok(status["openunit"])
    
    # Set sample interval
    us_for_block = ctypes.c_int32(1000)
    channels = ctypes.c_int32(drDaq.USB_DRDAQ_INPUTS["USB_DRDAQ_CHANNEL_TEMP"])
    no_of_channels = 1
    status["setInterval"] = drDaq.UsbDrDaqSetInterval(chandle, ctypes.byref(us_for_block), ideal_no_of_samples,
                                                      ctypes.byref(channels), no_of_channels)
    assert_pico_ok(status["setInterval"])
    
    # Find scaling information
    channel = drDaq.USB_DRDAQ_INPUTS["USB_DRDAQ_CHANNEL_TEMP"]
    nScales = ctypes.c_int16(0)
    currentScale = ctypes.c_int16(0)
    hoge=256
    names = (ctypes.c_char*hoge)()
    namesSize = hoge
    status["getscalings"] = drDaq.UsbDrDaqGetScalings(chandle, channel, ctypes.byref(nScales),
                                                      ctypes.byref(currentScale), ctypes.byref(names), namesSize)
    assert_pico_ok(status["getscalings"])
    
    print('AA ',nScales.value)
    print('BB ',currentScale.value)
    print('CC ',names.value)
    
    # Set channel scaling 
    scalingNumber = 0 #
    status["setscaling"] = drDaq.UsbDrDaqSetScalings(chandle, channel, scalingNumber)
    assert_pico_ok(status["setscaling"])
    
    # Set temperature compenstation
    #enabled = 1
    #status["TemperatureCompensation"] = drDaq.UsbDrDaqPhTemperatureCompensation(chandle, enabled)
    #assert_pico_ok(status["TemperatureCompensation"])

def get_data():
    global status
    global chandle
    # Run block capture
    method = drDaq.USB_DRDAQ_BLOCK_METHOD["BM_SINGLE"]
    status["run"] = drDaq.UsbDrDaqRun(chandle, ideal_no_of_samples, method)
    assert_pico_ok(status["run"])
    
    ready = ctypes.c_int16(0)
    
    while ready.value == 0:
        status["ready"] = drDaq.UsbDrDaqReady(chandle, ctypes.byref(ready))
        print(ready.value)
        time.sleep(0.1)
    
    # Retrieve data from device
    values = (ctypes.c_float * ideal_no_of_samples)()
    noOfValues = ctypes.c_uint32(ideal_no_of_samples)
    overflow = ctypes.c_uint16(0)
    triggerIndex = ctypes.c_uint32(0)
    status["getvaluesF"] = drDaq.UsbDrDaqGetValuesF(chandle, ctypes.byref(values), ctypes.byref(noOfValues),
                                                    ctypes.byref(overflow), ctypes.byref(triggerIndex))
    assert_pico_ok(status["getvaluesF"])
    
    # generate time data
    #time = np.linspace(0, us_for_block, ideal_no_of_samples)
    #now  = datetime.now()
    now  = time.time()
    temp = np.array(values[:]).mean()/10.0
    
    return now,temp

# plot the data
#plt.plot(time, values[:])
#plt.xlabel('Time (ns)')
#plt.ylabel('Temperature (degC)')
#plt.show()

def close():
    # Disconnect the scope
    # handle = chandle
    status["close"] = drDaq.UsbDrDaqCloseUnit(chandle)
    assert_pico_ok(status["close"])
    
    # Display status returns
    print(status)

def main():
    init() 
    ipts=0
    dateA=0
    tempA=0
    print('Start main loop')
    fig = plt.figure()
    with open(filename, 'w') as f:
        while(1):
            date,temp=get_data()
            arr=np.array([[date,temp]])
            print(arr)
            if ipts==0:
                dateA=np.array([date])
                tempA=np.array([temp])
            else:
                dateA=np.append(dateA,date)
                tempA=np.append(tempA,temp)
            ipts=ipts+1
            f.write(str(arr)+'\n')
            currentTemp = str(temp)
            #print(date,currentTemp)
            line = plt.plot(dateA,tempA,label=str(float('%.4g' % temp)))
            plt.xlabel('Datetime')
            plt.ylabel('Temperature')
            plt.ylim(18,28)
            plt.legend(loc='lower center',fontsize='x-large')
            plt.pause(interval)
            fig.clear()
            #time.sleep(interval)
    
    close() 
    plt.show()
    print('End')


#### main loop
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
