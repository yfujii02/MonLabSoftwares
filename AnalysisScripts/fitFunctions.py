import numpy as np
from scipy.stats import moyal, poisson

def gauss_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (( x - mu ) / sigma ) **2)

def pois_gauss_pdf(x, mu, plambda, gain, sigma0, sigma1, nlim):
    prob = Poisson(0,plambda) * gauss_pdf(x, mu, sigma0)
    for i in range(int(nlim)):
        prob = prob + Poisson(i+1,plambda) * gauss_pdf(x, mu+gain*(i+1), np.sqrt(sigma0*sigma0+(i+1)*sigma1*sigma1))
    return prob

def Gaus(x,height,centre,width):
    return height*np.exp(-(x - centre)**2 / (2*width**2))

def Poisson(k, lamb):
    return poisson.pmf(k, lamb)

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
