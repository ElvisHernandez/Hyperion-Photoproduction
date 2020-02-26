#!/bin/bash

for i in *.csv; do
    E=`cat $i | grep W | grep GEV | awk '{print $3}' | sed s/[][]// | sed s/[]]// | sed s/GEV// | sed s/,,,//`
    Observable=`cat $i | grep keyword | grep observables | awk '{print $4}'`
    if [ "$Observable" == "DSIG/DCOSTHETA" ]; then
        echo "File has correct observable"
	FileName="../../Sigma_Dey_E$E.raw"
	echo $i
	echo $FileName
	cat $i | sed '/#/d' | sed '/COS/d' | sed 's/,/ /g' > "$FileName"
    fi
done
