#!/bin/bash

rm out;
echo "NQUEEN 		SOLUTIONS 	BLOCKUSED 	QUEUEUSED 	MAXBLOCK 	MAXQUEUE 	L1 	L2 	TIME" > out;

START=3;
QUEEN=13;
BLOCK=10000;
QUEUE=5000;

for (( i=$START; i<$QUEEN; i++ )); do
	for (( j=$START; j<$QUEEN; j++ )); do
		if (($i <= $j)); then
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> out; 
			echo "done nQueen $QUEEN, max block $BLOCK, max queue $QUEUE, level1 $i, level2 $j";
		fi
	done;
done;
