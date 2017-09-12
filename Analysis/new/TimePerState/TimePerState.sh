NQUEEN=8;
n=$NQUEEN;
for((l=1;l<NQUEEN;l++)); do
	for((k=l+1;k<NQUEEN;k++)); do 
		for m in {1..5}; do
			../../../bin/RCUDA2 -n $NQUEEN -l $l -k $k -q 60000 -b 60000 -f >> out$n-$l-$k;
			echo "queens $NQUEEN level1 $j level2 $k"
		done;
	done;
done;

NQUEEN=10;
n=$NQUEEN;
for((l=0;l<1;l++)); do
	for((k=l+1;k<NQUEEN;k++)); do 
		for m in {1..5}; do
			../../../bin/RCUDA2 -n $NQUEEN -l $l -k $k -q 60000 -b 60000 -f >> out$n-$l-$k;
			echo "queens $NQUEEN level1 $j level2 $k"
		done;
	done;
done;


NQUEEN=12;
n=$NQUEEN;
for((l=1;l<NQUEEN;l++)); do
	for((k=l+1;k<NQUEEN;k++)); do 
		for m in {1..5}; do
			../../../bin/RCUDA2 -n $NQUEEN -l $l -k $k -q 60000 -b 60000 -f >> out$n-$l-$k;
			echo "queens $NQUEEN level1 $j level2 $k"
		done;
	done;
done;


spd-say done;
