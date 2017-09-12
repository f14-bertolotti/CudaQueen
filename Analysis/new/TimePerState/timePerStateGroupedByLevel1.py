from matplotlib import pyplot
from pylab import genfromtxt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go


QueenSets = [8,10,12]
sampleNumber = 5


meanTimes = []
meanBlocks = []
meanStates = []
timesPerState = []
pos = []
xTicksPos = [0]

for l in range(12):
	for QueenSet in QueenSets:
		for k in range(QueenSet-1):
			if l+1 < k+1 and l < QueenSet-1:
			 	matSet = genfromtxt("out"+str(QueenSet)+"-"+str(l+1)+"-"+str(k+1))
			 	times = matSet[:,9]
			 	blocks = matSet[:,3]
			 	states = matSet[:,2]

			 	sumTimes = 0
			 	sumBlocks = 0
			 	sumStates = 0
			 	for i in range(len(times)):
			 		sumTimes += times[i]
			 		sumBlocks += blocks[i]
			 		sumStates += states[i]
			 		if (i+1)%sampleNumber:
			 			meanTimes.append(sumTimes/sampleNumber)
			 			meanBlocks.append(sumBlocks/sampleNumber)
			 			meanStates.append(sumStates/sampleNumber)
			 			sumTimes = 0
			 			sumBlocks = 0
			 			sumStates = 0
			 			timesPerState.append(meanTimes[-1]/meanStates[-1])
			 			if not pos:
			 				pos = [0]
			 			else:
			 				pos.append(pos[-1]+1)
	if l < 10:
		for i in range(20):
			pos.append(pos[-1]+1)
			timesPerState.append(0)
		xTicksPos.append(pos[-1])

plt.xticks(xTicksPos,xrange(1,11))
plt.bar(pos,timesPerState,color='black')
plt.title("Grouped By Discriminant 1")
plt.xlabel('levelDiscriminant1');
plt.ylabel('time per state(ms)')
plt.savefig("GroupedByDiscriminant1.png")
plt.show()
#print timesPerState


