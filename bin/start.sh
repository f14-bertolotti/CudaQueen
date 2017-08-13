#!/bin/bash

rm out;
echo "NQUEEN SOLUTIONS NODES BLOCKUSED QUEUEUSED MAXBLOCK MAXQUEUE L1 L2 TIME" > stressedQ8;
echo "NQUEEN SOLUTIONS NODES BLOCKUSED QUEUEUSED MAXBLOCK MAXQUEUE L1 L2 TIME" > stressedQ10;

START=0;
QUEEN=10;
BLOCK=5000;
QUEUE=100;

for (( i=$START; i<$QUEEN; i++ )); do
	for (( j=$START; j<$QUEEN; j++ )); do
		if (($i < $j)); then
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> stressedQ10; 
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> stressedQ10; 
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> stressedQ10; 
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> stressedQ10; 
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> stressedQ10; 
		fi
	done;
done;

spd-say done;

START=0;
QUEEN=8;
BLOCK=5000;
QUEUE=10;

for (( i=$START; i<$QUEEN; i++ )); do
	for (( j=$START; j<$QUEEN; j++ )); do
		if (($i < $j)); then
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> stressedQ8; 
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> stressedQ8; 
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> stressedQ8; 
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> stressedQ8; 
			./RCUDA2 -n $QUEEN -b $BLOCK -q $QUEUE -l $i -k $j -f >> stressedQ8; 
		fi
	done;
done;
