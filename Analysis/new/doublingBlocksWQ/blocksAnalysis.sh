#!/bin/bash

NQUEEN=8;
n=$NQUEEN;
# with queue
for((j=1;j<NQUEEN;j++)); do
	for((k=j+1;k<NQUEEN;k++)); do
		for((t=NQUEEN;t<450;t=t*2)); do 
			for m in {1..5}; do
				../../../bin/RCUDA2 -n $NQUEEN -l $j -k $k -q 10000 -b $t -f >> out$n-$j-$k;
				echo "queens $NQUEEN level1 $j level2 $k"
			done;
		done;
	done;
done;

NQUEEN=10;
n=$NQUEEN;
# with queue
for((j=1;j<NQUEEN;j++)); do
	for((k=j+1;k<NQUEEN;k++)); do
		for((t=NQUEEN;t<6450;t=t*2)); do 
			for m in {1..5}; do
				../../../bin/RCUDA2 -n $NQUEEN -l $j -k $k -q 10000 -b $t -f >> out$n-$j-$k;
				echo "queens $NQUEEN level1 $j level2 $k"
			done;
		done;
	done;
done;

NQUEEN=12;
n=$NQUEEN;
# with queue
for((j=1;j<NQUEEN;j++)); do
	for((k=j+1;k<NQUEEN;k++)); do
		for((t=NQUEEN;t<10000;t=t*2)); do 
			for m in {1..5}; do
				../../../bin/RCUDA2 -n $NQUEEN -l $j -k $k -q 10000 -b $t -f >> out$n-$j-$k;
				echo "queens $NQUEEN level1 $j level2 $k"
			done;
		done;
	done;
done;


spd-say done;
