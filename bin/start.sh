#!/bin/bash

rm out;
echo "NQUEEN 		SOLUTIONS 	BLOCKUSED 	QUEUEUSED 	MAXBLOCK 	MAXQUEUE 	L1 	L2 	TIME" > out;

START=3;
QUEEN=12;
BLOCK=1000;
QUEUE=1000;

for (( i=$START; i<$QUEEN-3; i++ )); do
	for (( j=$START; j<$QUEEN-3; j++ )); do
		if (($i <= $j)); then
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f; 
		fi
	done;
done;
