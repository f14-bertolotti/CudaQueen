from matplotlib import pyplot
from pylab import genfromtxt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

nQueen = 8

matQ8  = genfromtxt("stressedQ8")
matQ10 = genfromtxt("stressedQ10")
matQn8  = genfromtxt("outQ8")
matQn10 = genfromtxt("outQ10")

timeQ8  = matQ8[:,9]
timeQ10 = matQ10[:,9]
timeQn8  = matQn8[:,9]
timeQn10 = matQn10[:,9]

i = 0
k = 0
sumQ8 = 0
sumQn8 = 0
meanQ8 = []
meanQn8 = []

for x in timeQ8[1:]:
	sumQ8 += timeQ8[i]
	sumQn8 += timeQn8[i]
	i += 1
	if i%5 == 0:
		k += 1
		sumQ8 = sumQ8/5
		sumQn8 = sumQn8/5
		meanQ8.append(sumQ8)
		meanQn8.append(sumQn8)
		sumQ8 = 0
		sumQn8 = 0

i = 0
k = 0
sumQ10 =0
sumQn10 = 0
meanQ10 = []
meanQn10 = []

for x in timeQ10[1:]:
	sumQ10 += timeQ10[i]
	sumQn10 += timeQn10[i]
	i += 1
	if i%5 == 0:
		k += 1
		sumQ10 = sumQ10/5
		sumQn10 = sumQn10/5
		meanQ10.append(sumQ10)
		meanQn10.append(sumQn10)
		sumQ10 = 0
		sumQn10 = 0


assiX8 = []
assiTicks8 = []
nQueen = 8
i = 0
for x in xrange(0,nQueen):
	for y in xrange(0,nQueen):
		if x < y:
			assiX8.append(i)
			assiTicks8.append(str(x)+","+str(y))
			i+=1;

assiX10 = []
assiTicks10 = []
nQueen = 10
i = 0
for x in xrange(0,nQueen):
	for y in xrange(0,nQueen):
		if x < y:
			assiX10.append(i)
			assiTicks10.append(str(x)+","+str(y))
			i+=1;

plt.subplot(2,1,1)
plt.xticks(assiX8,assiTicks8)
plt.plot(assiX8,meanQ8,color='blue')
plt.plot(assiX8,meanQn8,color='red')
plt.title('8 queen')
stressed = mpatches.Patch(color='blue', label='stresse q')
normal   = mpatches.Patch(color='red', label='normal q')
plt.legend(handles=[stressed, normal])
plt.grid(True)

plt.subplot(2,1,2)
plt.xticks(assiX10,assiTicks10)
plt.plot(assiX10,meanQ10,color='blue')
plt.plot(assiX10,meanQn10,color='red')
plt.title('10 queen')
stressed = mpatches.Patch(color='blue', label='stresse q')
normal   = mpatches.Patch(color='red', label='normal q')
plt.legend(handles=[stressed, normal])
plt.grid(True)
plt.show();

#plt.xticks(assiX,assiTicks)
#plt.plot(assiX,meanQ,linewidth=2.0,color='red',label="mean_with_q")
#plt.plot(assiX,meanNQ,linewidth=2.0,color='blue',label="mean_without_q")
#plt.plot(assiX,diff,linewidth=2.0,color='green',label="diff")
#Queue = mpatches.Patch(color='red', label='mean_with_q')
#NQueue = mpatches.Patch(color='blue', label='mean_without_q')
#Diff = mpatches.Patch(color='green', label='diff')
#plt.legend(handles=[Queue, NQueue, Diff]);
#plt.title('8 queen, q_ver 0')
#plt.grid(True)
#plt.show()





