#!/bin/bash

for arg;
do python3.7 InitDAQ.py $arg > tempfile.txt
done

#python3.7 RunDAQ.py Pico3000_settings.yaml > log3000.log &
#python3.7 RunDAQ.py Pico6000_settings.yaml > log6000.log &
