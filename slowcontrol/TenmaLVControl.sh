#!/bin/bash
##############################################################
# Shell script to ramp up the voltage for
# TENMA programmable DC voltage supply 72-2940
# This requirs the external package tenma-seral
# See more details in https://github.com/kxtells/tenma-serial
# Usage:
#   $ ./TenmaLVControl.sh $finalV $stepV1 $transitionV $stepV2 $up
#      ramp up/down if up is 1/-1
##############################################################
tenmaControl.py /dev/ttyACM0 # check module
#tenmaControl.py -v 0   /dev/ttyACM0 # make sure it's 0v
echo "maximum current set to 10 mA"
tenmaControl.py -c 10  /dev/ttyACM0 # current limit 10 mA
echo "runnig voltage:"
tenmaControl.py --runningVoltage /dev/ttyACM0
currentV=$( tenmaControl.py --runningVoltage /dev/ttyACM0 | tail -1)
echo "running voltage again: $currentV"
echo "print status:"
tenmaControl.py --status /dev/ttyACM0
tenmaControl.py --on   /dev/ttyACM0
####
# tenmaControl.py --off  /dev/ttyACM0
up=$5
if (( up > 0 )); then
    up=1
else
    up=-1
fi
targetV=$1
stepV1=$(echo "$up * $2" | bc -l)
transitionV=$3
stepV2=$(echo "$up * $4" | bc -l)
nstep1=$(echo "($transitionV-$currentV)/$stepV1" | bc)
nstep2=$(echo "($targetV-$transitionV)/$stepV2" | bc)
echo $nstep1 $nstep2
nstep=$(( $nstep1 + $nstep2 ))

function float_to_int(){
    echo $1 | cut -d. -f1
}

for (( i=0; i<$nstep; i++ ))
do
    if (( $(echo "$up * ($transitionV-$currentV) >= 0" |bc -l) ))
    then
        tempV=$(echo "$currentV + $stepV1" | bc -l)
    else
        tempV=$(echo "$currentV + $stepV2" | bc -l)
    fi
    currentV=$tempV
    inputV=$(float_to_int $(echo "1000*$currentV" | bc -l) )
    tenmaControl.py -v $inputV  /dev/ttyACM0 # make sure it's 0v
    #echo "$currentV $inputV"
    sleep 0.5
done
