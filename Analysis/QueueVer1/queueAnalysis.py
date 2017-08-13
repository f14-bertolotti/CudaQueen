from matplotlib import pyplot
from pylab import genfromtxt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

nQueen = 10

matQ = genfromtxt("outQ10")
matNQ = genfromtxt("outNQ10")

timeQ = matQ[:,9]
timeNQ = matNQ[:,9]

i = 0
k = 0
sumQ = 0
meanQ = []
sumNQ = 0
meanNQ = []
diff = []
for x in timeQ[1:]:
	sumNQ += timeNQ[i]
	sumQ += x
	i += 1
	if i%5 == 0:
		k += 1
		sumQ = sumQ/5
		sumNQ = sumNQ/5
		meanQ.append(sumQ)
		meanNQ.append(sumNQ)
		diff.append(sumQ-sumNQ)
		sumQ = 0
		sumNQ = 0

assiX = []
assiTicks = []
i = 0
for x in xrange(0,nQueen):
	for y in xrange(0,nQueen):
		if x < y:
			assiX.append(i)
			assiTicks.append(str(x)+","+str(y))
			i+=1;

plt.xticks(assiX,assiTicks)
plt.plot(assiX,meanQ,linewidth=2.0,color='red',label="mean_with_q")
plt.plot(assiX,meanNQ,linewidth=2.0,color='blue',label="mean_without_q")
plt.plot(assiX,diff,linewidth=2.0,color='green',label="diff")
Queue = mpatches.Patch(color='red', label='mean_with_q')
NQueue = mpatches.Patch(color='blue', label='mean_without_q')
Diff = mpatches.Patch(color='green', label='diff')
plt.legend(handles=[Queue, NQueue, Diff]);
plt.title('10 queen, q_ver 1')
plt.grid(True)
plt.show()
