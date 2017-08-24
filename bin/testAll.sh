#!/bin/bash

# rm out;

# with queue

for((i=0;i<=13;i++)); do
	for((j=0;j<i;j++)); do
		for((k=j+1;k<i;k++)); do
			./RCUDA2 -n $i -l $j -k $k -q 5000 -b 10000 -f >> /home/the14th/Dropbox/out;
			echo "queens $i level1 $j level2 $k"
		done;
	done;
done;

# without queue

for((i=0;i<=13;i++)); do
	for((j=0;j<i;j++)); do
		for((k=j+1;k<i;k++)); do
			./RCUDA2 -n $i -l $j -k $k -q 0 -b 10000 -f >> /home/the14th/Dropbox/out;
			echo "queens $i level1 $j level2 $k"
		done;
	done;
done;


spd-say done;