from matplotlib import pyplot
from pylab import genfromtxt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

mat1B = genfromtxt("1Block")
mat1BS = genfromtxt("1BlockSimple")
matNB = genfromtxt("NBlock")
matNBN = genfromtxt("NBlockNew")

times = 5

time1B = mat1B[:,9]
time1BS = mat1BS[:,9]
timeNB = matNB[:,9]
timeNBN = matNBN[:,9]
block = matNBN[:,3]

meanBlock = []
meanBlock2 = []
for x in range(len(block)):
 	if (x+1)%5 == 0:
 		meanBlock.append(x)
 		if x <= 416:
 			meanBlock2.append(x)
 		else:
 			meanBlock2.append(416)

print "len(time1BS)="+str(len(time1BS))
print "len(time1B) ="+str(len(time1B))
print "len(timeNB) ="+str(len(timeNB))
print "len(timeNBN)="+str(len(timeNBN))
print "len(block)  ="+str(len(block))

mean1BS = []
mean1B = []
meanNB = []
meanNBN = []
sum1BS = 0
sum1B = 0
sumNB = 0
sumNBN = 0
for x in range(len(time1BS)):
	sum1BS += time1BS[x]
	sum1B += time1B[x]
	sumNB += timeNB[x]
	sumNBN += timeNBN[x]
	if (x+1)%5==0:
		mean1BS.append(sum1BS/times)
		mean1B.append(sum1B/times)
		meanNB.append(sumNB/times)
		meanNBN.append(sumNBN/times)		
		sum1BS = 0
		sum1B = 0
		sumNB = 0
		sumNBN = 0

print "len(mean1BS)="+str(len(mean1BS))
print "len(mean1B) ="+str(len(mean1B))
print "len(meanNB) ="+str(len(meanNB))
print "len(meanNBN)="+str(len(meanNBN))
print "len(meanBlock)="+str(len(meanBlock))

plt.plot(range(4,13),mean1BS,linewidth=1.0,color='red',label="1BS")
plt.plot(range(4,13),mean1B,linewidth=1.0,color='blue',label="1B")
plt.plot(range(4,13),meanNB,linewidth=1.0,color='green',label="NB")
plt.plot(range(4,13),meanNBN,linewidth=1.0,color='magenta',label="NBN")

B = mpatches.Patch(color='red', label='1block')
BS = mpatches.Patch(color='blue', label='1blockSimple')
NB = mpatches.Patch(color='green', label='Nblock')
NBN = mpatches.Patch(color='magenta', label='NblockNew')

plt.legend(handles=[B, BS, NB, NBN],loc=2,prop={'size': 8});
plt.title('compare time versions')
plt.grid(True)
figure = plt.gcf()
figure.set_size_inches(14,12)
plt.savefig("compare.png", bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------

speedUp = []
efficiency = []

for x in range(len(mean1B)):
	speedUp.append(mean1B[x]/meanNBN[x])
 	efficiency.append(mean1B[x]/(meanNBN[x]*meanBlock[x]))

plt.bar(xrange(4,13),speedUp,linewidth=1.0,color='red',label="NBN vs 1B")

plt.title('1 block vs n block speed up')
plt.grid(True)
figure = plt.gcf()
figure.set_size_inches(14,12)
plt.savefig("speedUp.png")
plt.close()

#-----------------------------------------------------------------------------

plt.bar(xrange(4,13),efficiency,linewidth=1.0,color='red',label="NBN vs 1B")

plt.title('1 block vs n block efficiency')
plt.grid(True)
figure = plt.gcf()
figure.set_size_inches(14,12)
plt.savefig("efficiency.png")
plt.close()

#-----------------------------------------------------------------------------

speedUp = []

for x in range(len(mean1BS)):
	speedUp.append(mean1BS[x]/meanNBN[x])

plt.bar(xrange(4,13),speedUp,linewidth=1.0,color='red',label="NBN vs 1B")

plt.title('1 block Simple vs n block speed up')
plt.grid(True)
figure = plt.gcf()
figure.set_size_inches(14,12)
plt.savefig("speedUp2.png")
plt.close()



