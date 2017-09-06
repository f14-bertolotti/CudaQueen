#!/bin/bash

# rm out;

# with queue

for((i=0;i<=13;i++)); do
	for((j=0;j<i;j++)); do
		for((k=j+1;k<i;k++)); do
			./RCUDA -n $i -l $j -k $k -q 60000 -b 60000 -f >> /home/the14th/Dropbox/outWQ;
			echo "queens $i level1 $j level2 $k"
		done;
	done;
done;

# without queue

for((i=0;i<=13;i++)); do
	for((j=0;j<i;j++)); do

		./RCUDA -n $i -l $j -k 0 -q 0 -b 60000 -f >> /home/the14th/Dropbox/outNQ;
		echo "queens $i level1 $j level2 0"

	done;
done;


spd-say done;