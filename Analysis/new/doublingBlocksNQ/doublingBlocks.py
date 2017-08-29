from matplotlib import pyplot
from pylab import genfromtxt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

QueenSet1 = 8
QueenSet2 = 10
QueenSet3 = 12

sampleNumber = 5

for i in range(QueenSet1-1):
	matSet = genfromtxt("out"+str(QueenSet1)+"-"+str(i+1))
	print str(QueenSet1)+"-"+str(i+1)
	times = matSet[:,9]
	blocks = matSet[:,3]

	effs = []
	speeds = []
	meanTimes = []
	meanblocks = []
	xAss = [QueenSet1]
	assiXSet = []
	assiTicksSet = []

	sum = 0
	sum2 = 0
	for m in range(len(times)):
		sum += times[m]
		sum2 += blocks[m]
		if(m+1)%sampleNumber == 0:
			meanTimes.append(sum/sampleNumber)
			meanblocks.append(sum2/sampleNumber)
			sum = 0
			sum2 = 0 

	for l in range(len(meanTimes)-1):
		xAss.append(xAss[l]*2)

	for t in range(len(meanTimes)-1):
		speeds.append(meanTimes[t]/meanTimes[t+1]);
		effs.append(meanTimes[t]/(meanTimes[t+1]*(meanblocks[t+1]/meanblocks[t])));


	print meanTimes
	print meanblocks

	figure = plt.gcf()
	figure.set_size_inches(14,12)
	plt.subplot(2,1,1)
	plt.plot(meanblocks,meanTimes,color='red')
	plt.grid(True)
	plt.title(str(QueenSet1)+" queens and "+str(i+1)+" levels")
	plt.subplot(2,1,2)
	plt.plot(range(len(xAss)-1),speeds,color="blue")
	plt.plot(range(len(xAss)-1),effs,color="green")
	effPatch = mpatches.Patch(color="green",label="efficiency")
	speedPatch = mpatches.Patch(color="blue",label="speed up")
	plt.legend(handles=[effPatch,speedPatch],loc=2)
	plt.grid(True)
	plt.title("speed up and efficiency")
	plt.savefig("fig"+str(QueenSet1)+"-"+str(i+1))
	plt.close()

#---------------------------------------------------------------

for i in range(QueenSet2-1):
	matSet = genfromtxt("out"+str(QueenSet2)+"-"+str(i+1))
	print str(QueenSet2)+"-"+str(i+1)
	times = matSet[:,9]
	blocks = matSet[:,3]

	effs = []
	speeds = []
	meanTimes = []
	meanblocks = []
	xAss = [QueenSet2]
	assiXSet = []
	assiTicksSet = []

	sum = 0
	sum2 = 0
	for m in range(len(times)):
		sum += times[m]
		sum2 += blocks[m]
		if(m+1)%sampleNumber == 0:
			meanTimes.append(sum/sampleNumber)
			meanblocks.append(sum2/sampleNumber)
			sum = 0
			sum2 = 0 

	for l in range(len(meanTimes)-1):
		xAss.append(xAss[l]*2)

	for t in range(len(meanTimes)-1):
		speeds.append(meanTimes[t]/meanTimes[t+1]);
		effs.append(meanTimes[t]/(meanTimes[t+1]*(meanblocks[t+1]/meanblocks[t])));


	print meanTimes
	print meanblocks

	figure = plt.gcf()
	figure.set_size_inches(14,12)
	plt.subplot(2,1,1)
	plt.plot(meanblocks,meanTimes,color='red')
	plt.grid(True)
	plt.title(str(QueenSet2)+" queens and "+str(i+1)+" levels")
	plt.subplot(2,1,2)
	plt.plot(range(len(xAss)-1),speeds,color="blue")
	plt.plot(range(len(xAss)-1),effs,color="green")
	effPatch = mpatches.Patch(color="green",label="efficiency")
	speedPatch = mpatches.Patch(color="blue",label="speed up")
	plt.legend(handles=[effPatch,speedPatch],loc=2)
	plt.grid(True)
	plt.title("speed up and efficiency")
	plt.savefig("fig"+str(QueenSet2)+"-"+str(i+1))
	plt.close()

	#---------------------------------------------------------------

for i in range(QueenSet3-1):
	matSet = genfromtxt("out"+str(QueenSet3)+"-"+str(i+1))
	print str(QueenSet3)+"-"+str(i+1)
	times = matSet[:,9]
	blocks = matSet[:,3]

	effs = []
	speeds = []
	meanTimes = []
	meanblocks = []
	xAss = [QueenSet3]
	assiXSet = []
	assiTicksSet = []

	sum = 0
	sum2 = 0
	for m in range(len(times)):
		sum += times[m]
		sum2 += blocks[m]
		if(m+1)%sampleNumber == 0:
			meanTimes.append(sum/sampleNumber)
			meanblocks.append(sum2/sampleNumber)
			sum = 0
			sum2 = 0 

	for l in range(len(meanTimes)-1):
		xAss.append(xAss[l]*2)

	for t in range(len(meanTimes)-1):
		speeds.append(meanTimes[t]/meanTimes[t+1]);
		effs.append(meanTimes[t]/(meanTimes[t+1]*(meanblocks[t+1]/meanblocks[t])));


	print meanTimes
	print meanblocks

	figure = plt.gcf()
	figure.set_size_inches(14,12)
	plt.subplot(2,1,1)
	plt.plot(meanblocks,meanTimes,color='red')
	plt.grid(True)
	plt.title(str(QueenSet3)+" queens and "+str(i+1)+" levels")
	plt.subplot(2,1,2)
	plt.plot(range(len(xAss)-1),speeds,color="blue")
	plt.plot(range(len(xAss)-1),effs,color="green")
	effPatch = mpatches.Patch(color="green",label="efficiency")
	speedPatch = mpatches.Patch(color="blue",label="speed up")
	plt.legend(handles=[effPatch,speedPatch],loc=2)
	plt.grid(True)
	plt.title("speed up and efficiency")
	plt.savefig("fig"+str(QueenSet3)+"-"+str(i+1))
	plt.close()