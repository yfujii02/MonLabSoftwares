import numpy as np
from scipy.stats import moyal

def Gaus(x,height,centre,width):
    return height*np.exp(-(x - centre)**2 / (2*width**2))

####### Fitting with the four Gaussian functions with fixed intervals
def SPPeaksGaus4(x,a0,a1,a2,a3,s0,s1,s2,s3,m0,gain):
    val = 0
    ampl  = [a0,a1,a2,a3]
    sigma = [s0,s1,s2,s3]
    for i in range(4):
        val = val + Gaus(x, ampl[i],m0+i*gain,sigma[i])
    return val

###### 
def Moyal(x,*args):
    # args[0]: amplitude
    # args[1]: loc
    # args[2]: scale
    return args[0]*moyal.pdf(x, args[1],args[2])
