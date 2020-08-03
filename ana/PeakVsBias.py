import numpy as np
import matplotlib.pyplot as plt
#import DAQanalysis.py as daq


#Bias
V_Bias = 78 #Placeholder until I extract measurement from IV Curve

#Overvoltages for data 80.0-85.0 V
OverVoltage = [80.0, 81.0, 82.0, 83.0, 84.0, 85.0]
OverVoltage = np.array(OverVoltage)-V_Bias
print(OverVoltage)

#Peaks
ChAp = [32.86,49.96,51.86,53.76,74.65,101.24]
ChBp = [11.98,13.88,25.27,27.17,30.97,34.77]
ChCp = [7.28,9.34,12.20,15.76,15.76,20.66]
ChDp = [13.40,15.19,19.02,25.06,29.47,31.65]


plt.plot(OverVoltage, ChAp, color='blue')
plt.plot(OverVoltage, ChBp, color='red')
plt.plot(OverVoltage, ChCp, color='green')
plt.plot(OverVoltage, ChDp, color='yellow')
plt.xlabel("Overvoltage(V)")
plt.ylabel("Absolute Peak Distribution Voltage (mV)")
plt.title("Maximum Detector Peak vs Overvoltage")
plt.show()

