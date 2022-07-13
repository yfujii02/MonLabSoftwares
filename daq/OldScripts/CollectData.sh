#!/usr/bin/env bash

thre=6
events=1000
nsub=10
mode=1 # negative AND
for volt in 85.0 84.0 83.0 82.0 81.0 80.0 79.0 78.0 77.0;
do
  echo $volt $thre
  python3.7 ~/work/pico/daq/ScriptVoltageControl.py $volt
  sleep 1
  python3.7 runDAQ.py $events $nsub $mode Test${volt}V_Thre$thre $thre
  #sleep 1
done
python3.7 ~/work/pico/daq/ScriptVoltageControl.py 0

echo 'Data Collection Complete!'

