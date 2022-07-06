#Test Script for OWON/Multicomp USB Oscilloscope

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:03:26 2022

@author: smdek2
"""


# from vds1022 import *


# dev = VDS1022(debug=True)
# dev.set_timerange('20ms')
# dev.set_channel(CH1, range='50v', coupling='DC', offset='0v', probe='x10')

# for ch1, ch2 in dev.pull_iter(freq=1, autorange=True) :
#     print('Vrms:%s' % ch1.rms())

# import serial.tools.list_ports as port_list
# import usb

# ports = list(port_list.comports())
# for p in ports:
#     print(p)


# busses = usb.busses()
# for bus in busses:
#      devices = bus.devices
#      for dev in devices:
#           Dev = dev
# #         print("Device:", dev.filename)
# #         print("  idVendor: %d (0x%04x)" % (dev.idVendor, dev.idVendor))
# #         print("  idProduct: %d (0x%04x)" % (dev.idProduct, dev.idProduct))


# print("Device:", Dev.filename)
# print("  idVendor: %d (0x%04x)" % (Dev.idVendor, Dev.idVendor))
# print("  idProduct: %d (0x%04x)" % (Dev.idProduct, Dev.idProduct))

import usb.core
import usb.util
import sys

dev = usb.core.find(idVendor=0x5345,idProduct=0x1234)
print(dev.default_timeout)
if dev is None:
    raise ValueError('Device not found!')
else:

    # set the active configuration. With no arguments, the first
    # configuration will be the active one
    dev.set_configuration()
    print(dev)

msg = '*IDN?'

w = dev.write(3,msg)
print(w)
r=dev.read(0x81,len(msg))
print(r)

def send(cmd):
    #address taken from results of print(dev): ENDPOINT 0x3: Bulk OUT
    dev.write(3,cmd)
    #address taken from results of print(dev): ENDPOINT 0x81: Bulk IN
    result = (dev.read(0x81,100,10000))
    return result

def get_id():
    return send('*IDN?').tobytes().decode('utf-8')

def send_SCIPI():
    return send(':SDSLSCPI#')

def get_data(ch):
    #first 4 bytes indicate the number of data bytes following
    rawdata = send(':DATA:WAVE:SCREen:CH{}?'.format(ch))
    data = []
    for idx in range(4,len(rawdata),2):
       # take 2 bytes and convert them to signed integer using "little-endian"
       point = int().from_bytes([rawdata[idx], rawdata[idx+1]],'little',signed=True)
       data.append(point/4096)  # data as 12 bit
    return data

def get_header():
    # first 4 bytes indicate the number of data bytes following
    header = send(':DATA:WAVE:SCREen:HEAD?')
    header = header[4:].tobytes().decode('utf-8')
    return header

def save_data(ffname,data):
    f = open(ffname,'w')
    f.write('\n'.join(map(str, data)))
    f.close()

#print(send_SCIPI())
print(get_id())
header = get_header()
data = get_data(1)
save_data('Osci.dat',data)
### end of code

# # get an endpoint instance
# cfg = dev.get_active_configuration()
# intf = cfg[(0,0)]

# ep = usb.util.find_descriptor(
#     intf,
#     # match the first OUT endpoint
#     custom_match = \
#     lambda e: \
#         usb.util.endpoint_direction(e.bEndpointAddress) == \
#         usb.util.ENDPOINT_OUT)

# assert ep is not None

# for cfg in dev:
#     sys.stdout.write(str(cfg.bConfigurationValue) + '\n')
#     for intf in cfg:
#         sys.stdout.write('\t' + \
#                           str(intf.bInterfaceNumber) + \
#                           ',' + \
#                           str(intf.bAlternateSetting) + \
#                           '\n')
#         print(intf)
#         for ep in intf:
#             print(ep)
#             sys.stdout.write('\t\t' + \
#                               str(ep.bEndpointAddress) + \
#                               '\n') #Endpoint = 3?

# # write the data
# #ep.write('test')

# msg = 'test'
# assert len(dev.write(0x3,msg,100)) ==len(msg)
# ret = dev.read(0x81,len(msg),100)
# sret = ''.join([chr(x) for x in ret])
# assert sret == msg

