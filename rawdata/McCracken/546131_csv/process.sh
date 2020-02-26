#!/bin/bash

for i in *.csv; do
    E=`cat $i | grep W | grep GEV | awk '{print $3}' | sed s/[][]// | sed s/[]]// | sed s/GEV// | sed s/,,,//`
    #Polar=`cat $i | grep POL | awk '{print $5}' | sed s/[-,"'"]//g`
    Observable=`cat $i | grep keyword | grep observables | awk '{print $4}'`
    #echo $Polar
    #echo $E
    if [ "$Observable" == "DSIG/DCOSTHETA" ]; then
        echo "File has correct observable"
	FileName="../../Lambda_McCracken_E$E.raw"
	echo $i
	echo $FileName
	cat $i | sed '/#/d' | sed '/COS/d' | sed 's/,/ /g' > "$FileName"
    fi
    #if [ $i == "Table1.csv" ]; then
    #fi
done
