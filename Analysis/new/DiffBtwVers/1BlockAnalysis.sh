
for ((i=4;i<=12;i++)); do
	for m in {1..5}; do
		../../../bin/RCUDA2 -n $i -l 0 -k 0 -q 0 -b 1 -f >> 1BlockNew;
		echo "queens $i level1 0 level2 0"
	done;
done;

