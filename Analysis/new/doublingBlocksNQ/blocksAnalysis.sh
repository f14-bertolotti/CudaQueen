NQUEEN=8;
n=$NQUEEN;
# with queue
for((j=1;j<NQUEEN;j++)); do
	for((t=NQUEEN;t<513;t=t*2)); do 
		for m in {1..5}; do
			../../../bin/RCUDA2 -n $NQUEEN -l $j -k 0 -q 0 -b $t -f >> out$n-$j;
			echo "queens $NQUEEN level1 $j"
		done;
	done;
done;

NQUEEN=10;
n=$NQUEEN;
# with queue
for((j=1;j<NQUEEN;j++)); do
	for((t=NQUEEN;t<641;t=t*2)); do 
		for m in {1..5}; do
			../../../bin/RCUDA2 -n $NQUEEN -l $j -k 0 -q 0 -b $t -f >> out$n-$j;
			echo "queens $NQUEEN level1 $j"
		done;
	done;
done;

NQUEEN=12;
n=$NQUEEN;
# with queue
for((j=1;j<NQUEEN;j++)); do
	for((t=NQUEEN;t<768;t=t*2)); do 
		for m in {1..5}; do
			../../../bin/RCUDA2 -n $NQUEEN -l $j -k 0 -q 0 -b $t -f >> out$n-$j;
			echo "queens $NQUEEN level1 $j"
		done;
	done;
done;


spd-say done;
