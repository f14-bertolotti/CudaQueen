#!/bin/bash


for((i=4;i<=13;i++)); do
	for((j=0;j<=i;j++)); do
		
		./RCUDA2 -n $i -l $j -b 10000 -f;
	done;
done;

spd-say done;
