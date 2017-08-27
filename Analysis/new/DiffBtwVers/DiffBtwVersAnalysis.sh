
for m in {1..5}; do
	../../../bin/RCUDA2 -n 4 -l 1 -k 3 -q 60000 -b 60000 -f >> NBlockNew;
	echo "queens $NQUEEN level1 $j level2 $k"
done;

for m in {1..5}; do
	../../../bin/RCUDA2 -n 5 -l 1 -k 3 -q 60000 -b 60000 -f >> NBlockNew;
	echo "queens $NQUEEN level1 $j level2 $k"
done;

for m in {1..5}; do
	../../../bin/RCUDA2 -n 6 -l 1 -k 3 -q 60000 -b 60000 -f >> NBlockNew;
	echo "queens $NQUEEN level1 $j level2 $k"
done;

for m in {1..5}; do
	../../../bin/RCUDA2 -n 7 -l 2 -k 4 -q 60000 -b 60000 -f >> NBlockNew;
	echo "queens $NQUEEN level1 $j level2 $k"
done;

for m in {1..5}; do
	../../../bin/RCUDA2 -n 8 -l 3 -k 5 -q 60000 -b 60000 -f >> NBlockNew;
	echo "queens $NQUEEN level1 $j level2 $k"
done;

for m in {1..5}; do
	../../../bin/RCUDA2 -n 9 -l 3 -k 6 -q 60000 -b 60000 -f >> NBlockNew;
	echo "queens $NQUEEN level1 $j level2 $k"
done;

for m in {1..5}; do
	../../../bin/RCUDA2 -n 10 -l 3 -k 7 -q 60000 -b 60000 -f >> NBlockNew;
	echo "queens $NQUEEN level1 $j level2 $k"
done;

for m in {1..5}; do
	../../../bin/RCUDA2 -n 11 -l 4 -k 8 -q 60000 -b 60000 -f >> NBlockNew;
	echo "queens $NQUEEN level1 $j level2 $k"
done;

for m in {1..5}; do
	../../../bin/RCUDA2 -n 12 -l 4 -k 9 -q 60000 -b 60000 -f >> NBlockNew;
	echo "queens $NQUEEN level1 $j level2 $k"
done;

for m in {1..5}; do
	../../../bin/RCUDA2 -n 13 -l 4 -k 10 -q 60000 -b 60000 -f >> NBlockNew;
	echo "queens $NQUEEN level1 $j level2 $k"
done;

