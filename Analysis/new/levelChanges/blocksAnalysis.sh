NQUEEN=4
n=$NQUEEN;
for((j=0;j<NQUEEN;j++)); do
	for m in {1..5}; do
		../../../bin/RCUDA2 -n $NQUEEN -l $j -k 0 -q 0 -b 60000 -f >> out$n;
		echo "queens $NQUEEN level1 $j"
	done;
done;


NQUEEN=6
n=$NQUEEN;
for((j=0;j<NQUEEN;j++)); do
	for m in {1..5}; do
		../../../bin/RCUDA2 -n $NQUEEN -l $j -k 0 -q 0 -b 60000 -f >> out$n;
		echo "queens $NQUEEN level1 $j"
	done;
done;


NQUEEN=8;
n=$NQUEEN;
for((j=0;j<NQUEEN;j++)); do
	for m in {1..5}; do
		../../../bin/RCUDA2 -n $NQUEEN -l $j -k 0 -q 0 -b 60000 -f >> out$n;
		echo "queens $NQUEEN level1 $j"
	done;
done;

NQUEEN=10;
n=$NQUEEN;
for((j=0;j<NQUEEN;j++)); do
	for m in {1..5}; do
		../../../bin/RCUDA2 -n $NQUEEN -l $j -k 0 -q 0 -b 60000 -f >> out$n;
		echo "queens $NQUEEN level1 $j"
	done;
done;

NQUEEN=12;
n=$NQUEEN;
for((j=0;j<NQUEEN;j++)); do
	for m in {1..5}; do
		../../../bin/RCUDA2 -n $NQUEEN -l $j -k 0 -q 0 -b 60000 -f >> out$n;
		echo "queens $NQUEEN level1 $j"
	done;
done;


spd-say done;
