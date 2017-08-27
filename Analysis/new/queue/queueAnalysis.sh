#!/bin/bash

NQUEEN=8;

# with queue
for((j=0;j<NQUEEN;j++)); do
	for((k=j+1;k<NQUEEN;k++)); do
		for t in {1..5}; do 
			../../../bin/RCUDA2 -n $NQUEEN -l $j -k $k -q 1000 -b 1000 -f >> outWQ8;
			echo "queens $NQUEEN level1 $j level2 $k"
		done;
	done;
done;
# withouth queue
for((j=0;j<NQUEEN;j++)); do
	for((k=j+1;k<NQUEEN;k++)); do
		for t in {1..5}; do 
			../../../bin/RCUDA2 -n $NQUEEN -l $j -k $k -q 0 -b 1000 -f >> outNQ8;
			echo "queens $NQUEEN level1 $j level2 $k"
		done;
	done;
done;

NQUEEN=10;

# with queue
for((j=0;j<NQUEEN;j++)); do
	for((k=j+1;k<NQUEEN;k++)); do
		for t in {1..5}; do 
			../../../bin/RCUDA2 -n $NQUEEN -l $j -k $k -q 1000 -b 1000 -f >> outWQ8;
			echo "queens $NQUEEN level1 $j level2 $k"
		done;
	done;
done;
# withouth queue
for((j=0;j<NQUEEN;j++)); do
	for((k=j+1;k<NQUEEN;k++)); do
		for t in {1..5}; do 
			../../../bin/RCUDA2 -n $NQUEEN -l $j -k $k -q 0 -b 1000 -f >> outNQ8;
			echo "queens $NQUEEN level1 $j level2 $k"
		done;
	done;
done;

NQUEEN=12;

# with queue
for((j=0;j<NQUEEN;j++)); do
	for((k=j+1;k<NQUEEN;k++)); do
		for t in {1..5}; do 
			../../../bin/RCUDA2 -n $NQUEEN -l $j -k $k -q 1000 -b 1000 -f >> outWQ8;
			echo "queens $NQUEEN level1 $j level2 $k"
		done;
	done;
done;
# withouth queue
for((j=0;j<NQUEEN;j++)); do
	for((k=j+1;k<NQUEEN;k++)); do
		for t in {1..5}; do 
			../../../bin/RCUDA2 -n $NQUEEN -l $j -k $k -q 0 -b 1000 -f >> outNQ8;
			echo "queens $NQUEEN level1 $j level2 $k"
		done;
	done;
done;


spd-say done;
