from matplotlib import pyplot
from pylab import genfromtxt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

QueenSet1 = 8
QueenSet2 = 10
QueenSet3 = 12

maxDiff8 = 0
maxDiff10 = 0
maxDiff12 = 0

sampleNumber = 5

matWQSet1 = genfromtxt("outWQ8")
matNQSet1 = genfromtxt("outNQ8")
matWQSet2 = genfromtxt("outWQ10")
matNQSet2 = genfromtxt("outNQ10")
matWQSet3 = genfromtxt("outWQ12")
matNQSet3 = genfromtxt("outNQ12")

timeWQSet1 = matWQSet1[:,9]
timeNQSet1 = matNQSet1[:,9]
timeWQSet2 = matWQSet2[:,9]
timeNQSet2 = matNQSet2[:,9]
timeWQSet3 = matWQSet3[:,9]
timeNQSet3 = matNQSet3[:,9]

print "lenghts:"
print "set1 with queue:    "+str(len(timeWQSet1))
print "set1 without queue: "+str(len(timeNQSet1))
print "set2 with queue:    "+str(len(timeWQSet2))
print "set2 without queue: "+str(len(timeNQSet2))
print "set3 with queue:    "+str(len(timeWQSet3))
print "set3 without queue: "+str(len(timeNQSet3))

sumWQTemp = 0;
sumNQTemp = 0;
meanWQSet1 = [];
meanNQSet1 = [];

for i in range(len(timeWQSet1)):
	sumWQTemp += timeWQSet1[i]
	sumNQTemp += timeNQSet1[i]
	print str(timeNQSet1[i])+"	"+str(timeWQSet1[i])
	if (i+1)%sampleNumber == 0:
		meanWQSet1.append(sumWQTemp/sampleNumber)
		meanNQSet1.append(sumNQTemp/sampleNumber)
		sumWQTemp  = 0
		sumNQTemp = 0

sumWQTemp = 0;
sumNQTemp = 0;
meanWQSet2 = [];
meanNQSet2 = [];

for i in range(len(timeWQSet2)):
	sumWQTemp += timeWQSet2[i]
	sumNQTemp += timeNQSet2[i]
	if (i+1)%sampleNumber == 0:
		meanWQSet2.append(sumWQTemp/sampleNumber)
		meanNQSet2.append(sumNQTemp/sampleNumber)
		sumWQTemp  = 0
		sumNQTemp = 0

sumWQTemp = 0;
sumNQTemp = 0;
meanWQSet3 = [];
meanNQSet3 = [];


for i in range(len(timeWQSet3)):
	sumWQTemp += timeWQSet3[i]
	sumNQTemp += timeNQSet3[i]
	if (i+1)%sampleNumber == 0:
		meanWQSet3.append(sumWQTemp/sampleNumber)
		meanNQSet3.append(sumNQTemp/sampleNumber)
		sumWQTemp  = 0
		sumNQTemp = 0


assiXSet1 = []
assiTicksSet1 = []
count = 0
for i in range(QueenSet1):
 	for j in range(QueenSet1):
 		if i < j:
 			assiXSet1.append(count)
 			assiTicksSet1.append(str(i)+","+str(j))
 			count += 1;

assiXSet2 = []
assiTicksSet2 = []
count = 0
for i in range(QueenSet2):
 	for j in range(QueenSet2):
 		if i < j:
 			assiXSet2.append(count)
 			assiTicksSet2.append(str(i)+","+str(j))
 			count += 1;

assiXSet3 = []
assiTicksSet3 = []
count = 0
for i in range(QueenSet3):
 	for j in range(QueenSet3):
 		if i < j:
 			assiXSet3.append(count)
 			assiTicksSet3.append(str(i)+","+str(j))
 			count += 1;

for i in range(len(meanWQSet1)):
	if(meanNQSet1[i]-meanWQSet1[i]>maxDiff8):
		maxDiff8 = meanNQSet1[i]-meanWQSet1[i];

for i in range(len(meanWQSet2)):
	if(meanNQSet2[i]-meanWQSet2[i]>maxDiff10):
		maxDiff10 = meanNQSet2[i]-meanWQSet2[i];

for i in range(len(meanWQSet3)):
	if(meanNQSet3[i]-meanWQSet3[i]>maxDiff12):
		maxDiff12 = meanNQSet3[i]-meanWQSet3[i];

print "max Diff 8: " + str(maxDiff8)
print "max Diff 10: " + str(maxDiff10)
print "max Diff 12: " + str(maxDiff12)


figure = plt.gcf()
figure.set_size_inches(16,12)
plt.xticks(assiXSet1,assiTicksSet1)
plt.plot(assiXSet1,meanWQSet1,linewidth=1.0,color='red',label="mean_with_q")
plt.plot(assiXSet1,meanNQSet1,linewidth=1.0,color='blue',label="mean_without_q")
WQueue = mpatches.Patch(color='red', label='with queue')
NQueue = mpatches.Patch(color='blue', label='withouth queue')
plt.legend(handles=[WQueue, NQueue]);
plt.xlabel('n. queens')
plt.ylabel('times(ms)')
plt.title('8 queen')
plt.grid(True)
plt.savefig('8queen.png', bbox_inches='tight');
plt.close()

figure = plt.gcf()
figure.set_size_inches(16,12)
plt.xticks(assiXSet2,assiTicksSet2)
plt.plot(assiXSet2,meanWQSet2,linewidth=1.0,color='red',label="mean_with_q")
plt.plot(assiXSet2,meanNQSet2,linewidth=1.0,color='blue',label="mean_without_q")
WQueue = mpatches.Patch(color='red', label='with queue')
NQueue = mpatches.Patch(color='blue', label='withouth queue')
plt.legend(handles=[WQueue, NQueue]);
plt.title('10 queen')
plt.xlabel('<levelDiscriminant1,leveldiscriminant2>')
plt.ylabel('times(ms)')
plt.grid(True)
plt.savefig('10queen.png', bbox_inches='tight');
plt.close()

figure = plt.gcf()
figure.set_size_inches(16,12)
plt.xticks(assiXSet3,assiTicksSet3)
plt.plot(assiXSet3,meanWQSet3,linewidth=1.0,color='red',label="mean_with_q")
plt.plot(assiXSet3,meanNQSet3,linewidth=1.0,color='blue',label="mean_without_q")
WQueue = mpatches.Patch(color='red', label='with queue')
NQueue = mpatches.Patch(color='blue', label='withouth queue')
plt.legend(handles=[WQueue, NQueue]);
plt.title('12 queen')
plt.xlabel('n. queens')
plt.ylabel('times(ms)')
plt.grid(True)
plt.savefig('12queen.png', bbox_inches='tight');
plt.close()
