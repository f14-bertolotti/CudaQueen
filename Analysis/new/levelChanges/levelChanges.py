from matplotlib import pyplot
from pylab import genfromtxt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

QueenSets = [4,6,8,10,12]
sampleNumber = 5


for QueenSet in QueenSets:
	matSet = genfromtxt("out"+str(QueenSet))
	print "out"+str(QueenSet)
	times = matSet[:,9]
	blocks = matSet[:,3]
	effs = []
	speeds = []
	meanTimes = []
	meanblocks = []

	sum = 0
	sum2 = 0
	for t in range(len(times)):
		sum += times[t]
		sum2 += blocks[t]
		if(t+1)%sampleNumber == 0:
			meanTimes.append(sum/sampleNumber)
			meanblocks.append(sum2/sampleNumber)
			sum = 0
			sum2 = 0 

	for t in range(len(meanTimes)-1):
		speeds.append(meanTimes[t]/meanTimes[t+1]);
		effs.append(meanTimes[t]/(meanTimes[t+1]*(meanblocks[t+1]/meanblocks[t])));

	assiXSet1 = []
	assiTicksSet1 = []
	for t in range(len(meanTimes)-1):
		assiXSet1.append(t)
		assiTicksSet1.append(str(int(meanblocks[t]))+"-"+str(int(meanblocks[t+1])))

	assiXSet2 = []
	assiTicksSet2 = []
	for t in range(len(meanTimes)):
		assiXSet2.append(t)
		assiTicksSet2.append(str(int(meanblocks[t])))


	print meanblocks
	print meanTimes
	print assiTicksSet1

	figure = plt.gcf()
	figure.set_size_inches(16,12)

	plt.subplot(3,1,1)
	plt.xticks(assiXSet2,assiTicksSet2)
	plt.bar(assiXSet2,meanTimes,color='black')
	plt.grid(True)
	plt.title(str(QueenSet)+" queens")

	plt.subplot(3,1,2)
	plt.xticks(assiXSet1,assiTicksSet1)
	plt.bar(assiXSet1,speeds,color="black")
	plt.title("speed up")
	plt.grid(True)


	plt.subplot(3,1,3)
	plt.xticks(assiXSet1,assiTicksSet1)
	plt.bar(assiXSet1,effs,color="black")
	plt.title("efficiency")
	plt.grid(True)

	plt.savefig("fig"+str(QueenSet))
	plt.close()
