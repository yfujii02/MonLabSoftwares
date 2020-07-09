import sys
import numpy as np
from scipy import signal as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

args=sys.argv

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N])/ float(N)

def peakSearch_simple(wf,height,width,start,nwindow):
    nabove=0
    peak=[]
    for i in range(start,start+nwindow):
        if wf[i]>height: nabove=nabove+1
        else:
            if nabove>=width:
                maximum=-1
                peakidx=-1
                for j in range(i,i-nabove,-1):
                    if maximum<wf[j]:
                        maximum=wf[j]
                        peakidx=j
                peak.append(peakidx)
            nabove=0
    return np.array(peak,dtype=int)

def peakSearch_ss(wf,height,width):
    peaks, _ = ss.find_peaks(wf,height,width)
    return peaks

fname=args[1]

startTime=125
print(startTime*0.8)
windowSize=int(float(args[2])/0.8)
nsub=int(args[3])
#pedRegion=startTime-20
pedRegion=int(45./0.8)
numAve=5
startPoint = int(startTime/0.8)
startPoint2= startPoint + numAve

flist=[]
if fname.find(".txt")<0:
    for f in range(nsub):
        flist.append(fname+str(f)+'.npy')
else:
    f=open(fname,'r')
    line=f.readline()
    while (line!=''):
        line=f.readline()
        if line.rstrip()=='':break
        flist.append(line.rstrip())
    f.close()
if nsub>len(flist):
    nsub=len(flist)

plt.ion()
plt.figure(figsize=(10,4))
base=np.array([])
wfrms=np.array([])
vmax=np.array([])
charge=np.array([])
polarity=-1.0
timeIntervalns=0.8
startPeakSearch=int(120./0.8)
thr=0.45

ntot=0
for f in range(nsub):
    ifile=open(flist[f],"rb")
    arrays=np.load(ifile,allow_pickle=True)
    nev=len(arrays)
    for i in range(nev):
        arr=arrays[i]
        avwf = running_mean(arr[:],numAve)
        #sos = ss.butter(2,[1e6,400e6],btype='bandpass',analog=False,fs=2.5e9,output='sos')
        #avwf = ss.sosfilt(sos,arr)
        if avwf[:pedRegion].std() > thr: continue
        #baseline =avwf[:windowSize].mean()
        #chargeTmp=(polarity*(avwf[startPoint:startPoint+windowSize].sum()-baseline*windowSize))
        #wfrms = np.append(wfrms, avwf[:windowSize].std())     # standard deviation
        baseline =avwf[:pedRegion].mean()
        avwf = polarity*(avwf-baseline)
        #peaks = peakSearch_ss(avwf[startPeakSearch:],0.7,5)
        peaks = peakSearch_simple(avwf,0.75,10,startPoint2,windowSize)
        chargeTmp=avwf[startPoint:startPoint+windowSize].sum()
        base  = np.append(base, baseline)                    # mean value of the baseline
        wfrms = np.append(wfrms, avwf[:pedRegion].std())     # standard deviation
        #peaks=peaks+startPeakSearch
        if len(peaks)>0:
            vmax  = np.append(vmax,avwf[peaks[0]])
            charge= np.append(charge,(avwf[peaks[0]-8:peaks[0]+52].sum())*timeIntervalns)   # integrated charge
            prePeak=peaks[0]
        timeX = np.linspace(0, float(len(arr))*timeIntervalns, len(arr))
        #print(arr)
        ntot = ntot+1
        if(i%200!=0):continue
        plt.plot(timeX,-arr)
        plt.plot(timeX[numAve-1:],avwf)
        plt.plot(timeX[peaks+numAve],avwf[peaks],"x")
        #plt.plot(timeX,avwf)
        #plt.plot(timeX[peaks],avwf[peaks],"x")
        plt.ylim(-4,10)
        plt.ylabel('voltage [mV]')
        plt.xlabel('time [ns]')
        plt.draw()
        plt.pause(0.1)
        plt.clf()
    
    ifile.close

nbins=120

fig = plt.figure(figsize=(10,8))
gs  = gridspec.GridSpec(2,2,hspace=0.35)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[1,0])
ax4 = plt.subplot(gs[1,1])

ymax=0.07*float(ntot)
xbins=np.linspace(0,0.65,nbins)
ax1.hist(wfrms,bins=xbins)
ax1.set_ylim(0.8,ymax)
ax1.set_title('Baseline RMS')
#ax1.text(+0.5,0.4*ymax,r'$\mu=$'+f'{wfrms.mean():.2f}'+' mV',fontsize=12)
ax1.set_xlabel('rms (mv)')

ymax=0.07*float(ntot)
xbins=np.linspace(0,8,nbins)
ax2.hist(vmax,bins=xbins)
ax2.set_ylim(0.8,ymax)
#ax2.set_yscale('log')
#ax2.text(4.0,0.4*ymax,r'$\mu=$'+f'{vmax.mean():.2f}'+' mV',fontsize=12)
ax2.set_title('Min Peak value')
ax2.set_xlabel('Voltage (mv)')

ymax=0.04*float(ntot)
xbins=np.linspace(0,120,nbins)
ax3.hist(charge,bins=xbins)
ax3.set_ylim(0.8,ymax)
#ax3.set_yscale('log')
#ax3.text(22,0.4*ymax,r'$\mu=$'+f'{charge.mean():.2f}',fontsize=12)
ax3.set_title('Integrated charge')
ax3.set_xlabel('Charge (mv*ns)')

fig.savefig('/home/comet/Desktop/figC.png')

#plt.clear()
