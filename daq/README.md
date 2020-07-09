- Run DAQ
```
$ python3 runDAQ.py [Number of events / subrun] [Number of subruns] [DAQ mode] [Output file name] [Threshold in mV]
```
- Output file from DAQ
HEADER of one subrun file.
|"BEGINHEADER",chStatus,daqStartTime,daqEndTime,nEvents,maxSamples|
Followed by waveforms (only read enabled channels are to be saved)
= (Number of enabled channels) x (nEvents) = (Number of total saved waveforms)
[Structure]
|Waveform ch1|Waveform ch2|Waveform ch3|Waveform ch4|
...
...
...
|Waveform ch1|Waveform ch2|Waveform ch3|Waveform ch4|
