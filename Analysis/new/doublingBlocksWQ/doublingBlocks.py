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
	for j in range(QueenSet1-1):
		if i+1 < j+1:
			matSet = genfromtxt("out"+str(QueenSet1)+"-"+str(i+1)+"-"+str(j+1))
			print str(QueenSet1)+"-"+str(i+1)+"-"+str(j+1)
			times = matSet[:,9]
			blocks = matSet[:,3]
			queues = matSet[:,4]

			effs = []
			speeds = []
			assiXSet = []
			assiTicksSet = []

			for t in range(len(blocks)):
	 			assiXSet.append(t)
	 			assiTicksSet.append(str(blocks[t])+","+str(queues[t]))

			for t in range(len(times)-1):
				speeds.append(times[t]/times[t+1]);
				effs.append(times[t]/(times[t+1]*(blocks[t+1]/blocks[t])));

			figure = plt.gcf()
			figure.set_size_inches(14,12)
			plt.subplot(2,1,1)
			plt.xticks(assiXSet,assiTicksSet)
			plt.plot(assiXSet,times,color='red')
			plt.grid(True)
			plt.title(str(QueenSet1)+" queens and "+str(i+1)+"-"+str(j+1)+"levels")
			plt.subplot(2,1,2)
			plt.plot(range(len(speeds)),speeds,color="blue")
			plt.plot(range(len(speeds)),effs,color="green")
			effPatch = mpatches.Patch(color="green",label="efficiency")
			speedPatch = mpatches.Patch(color="blue",label="speed up")
			plt.legend(handles=[effPatch,speedPatch],loc=2)
			plt.grid(True)
			plt.title("speed up and efficiency")
			plt.savefig("fig"+str(QueenSet1)+"-"+str(i+1)+"-"+str(j+1), bbox_inches='tight')
			plt.close()

for i in range(QueenSet2-1):
	for j in range(QueenSet2-1):
		if i+1 < j+1:
			matSet = genfromtxt("out"+str(QueenSet2)+"-"+str(i+1)+"-"+str(j+1))
			print str(QueenSet2)+"-"+str(i+1)+"-"+str(j+1)
			times = matSet[:,9]
			blocks = matSet[:,3]
			queues = matSet[:,4]

			effs = []
			speeds = []
			assiXSet = []
			assiTicksSet = []

			for t in range(len(blocks)):
	 			assiXSet.append(t)
	 			assiTicksSet.append(str(blocks[t])+","+str(queues[t]))

			for t in range(len(times)-1):
				speeds.append(times[t]/times[t+1]);
				effs.append(times[t]/(times[t+1]*(blocks[t+1]/blocks[t])));

			figure = plt.gcf()
			figure.set_size_inches(14,12)
			plt.subplot(2,1,1)
			plt.xticks(assiXSet,assiTicksSet)
			plt.plot(assiXSet,times,color='red')
			plt.grid(True)
			plt.title(str(QueenSet2)+" queens and "+str(i+1)+"-"+str(j+1)+"levels")
			plt.subplot(2,1,2)
			plt.plot(range(len(speeds)),speeds,color="blue")
			plt.plot(range(len(speeds)),effs,color="green")
			effPatch = mpatches.Patch(color="green",label="efficiency")
			speedPatch = mpatches.Patch(color="blue",label="speed up")
			plt.legend(handles=[effPatch,speedPatch],loc=2)
			plt.grid(True)
			plt.title("speed up and efficiency")
			plt.savefig("fig"+str(QueenSet2)+"-"+str(i+1)+"-"+str(j+1), bbox_inches='tight')
			plt.close()

for i in range(QueenSet3-1):
	for j in range(QueenSet3-1):
		if i+1 < j+1:
			matSet = genfromtxt("out"+str(QueenSet3)+"-"+str(i+1)+"-"+str(j+1))
			print str(QueenSet3)+"-"+str(i+1)+"-"+str(j+1)
			times = matSet[:,9]
			blocks = matSet[:,3]
			queues = matSet[:,4]

			effs = []
			speeds = []
			assiXSet = []
			assiTicksSet = []

			for t in range(len(blocks)):
	 			assiXSet.append(t)
	 			assiTicksSet.append(str(blocks[t])+","+str(queues[t]))

			for t in range(len(times)-1):
				speeds.append(times[t]/times[t+1]);
				effs.append(times[t]/(times[t+1]*(blocks[t+1]/blocks[t])));

			figure = plt.gcf()
			figure.set_size_inches(14,12)
			plt.subplot(2,1,1)
			plt.xticks(assiXSet,assiTicksSet)
			plt.plot(assiXSet,times,color='red')
			plt.grid(True)
			plt.title(str(QueenSet3)+" queens and "+str(i+1)+"-"+str(j+1)+"levels")
			plt.subplot(2,1,2)
			plt.plot(range(len(speeds)),speeds,color="blue")
			plt.plot(range(len(speeds)),effs,color="green")
			effPatch = mpatches.Patch(color="green",label="efficiency")
			speedPatch = mpatches.Patch(color="blue",label="speed up")
			plt.legend(handles=[effPatch,speedPatch],loc=2)
			plt.grid(True)
			plt.title("speed up and efficiency")
			plt.savefig("fig"+str(QueenSet3)+"-"+str(i+1)+"-"+str(j+1), bbox_inches='tight')
			plt.close()
