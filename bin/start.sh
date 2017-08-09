#!/bin/bash

START=0;
END=1;

rm out;
echo "NQUEEN 		SOLUTIONS 	BLOCKUSED 	MAXBLOCK 	TIME" > out;

for i in {0..4}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 4 -b 60000 -l $i -f >> out; 
		echo "done nQueen 4, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;

for i in {0..5}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 5 -b 60000 -l $i -f >> out; 
		echo "done nQueen 5, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;

for i in {0..6}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 6 -b 60000 -l $i -f >> out; 
		echo "done nQueen 6, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;

for i in {0..7}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 7 -b 60000 -l $i -f >> out; 
		echo "done nQueen 7, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;

for i in {0..8}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 8 -b 60000 -l $i -f >> out; 
		echo "done nQueen 8, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;

for i in {0..9}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 9 -b 60000 -l $i -f >> out; 
		echo "done nQueen 9, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;

for i in {0..10}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 10 -b 60000 -l $i -f >> out; 
		echo "done nQueen 10, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;

for i in {1..11}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 11 -b 60000 -l $i -f >> out; 
		echo "done nQueen 11, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;

for i in {2..12}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 12 -b 60000 -l $i -f >> out; 
		echo "done nQueen 12, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;

for i in {3..13}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 13 -b 60000 -l $i -f >> out; 
		echo "done nQueen 13, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;

for i in {3..14}; do
	for (( j=$START; j<$END; j++ )); do
		./RCUDA2 -n 14 -b 60000 -l $i -f >> out; 
		echo "done nQueen 14, max block 60000, level discriminant1 $i per $j of $END"
	done;
done;