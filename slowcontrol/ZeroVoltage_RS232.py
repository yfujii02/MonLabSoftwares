# -*- coding: utf-8 -*-
"""
Created on Wed 22/09/21
Last Updated: Wed 22/09/21
Author: Sam Dekkers

Run to safely zero Keithley 6487 Picoammeter via RS232 connection if there
has been some kind of error leaving it at a high voltage
"""
import sys
import VoltageFunctions as VF

args = sys.argv
numberargs = 2

VF.InitParameters(0,float(args[1]),0.2,2.0,0,0)
VF.ZeroVoltage(VF.ReadVoltage())
VF.Clear()
