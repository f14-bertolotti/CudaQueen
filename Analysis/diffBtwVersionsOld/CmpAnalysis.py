from matplotlib import pyplot
from pylab import genfromtxt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

mat12 = genfromtxt("1Block")
mat13 = genfromtxt("1BlockSimple")
mat14 = genfromtxt("NBlock")

time12 = mat12[1:,9]
time13 = mat13[1:,9]
time14 = mat14[1:,9]
block = mat14[:,3]

i = 0
meanBlock = []
meanBlock2 = []
for x in block:
	i += 1
	if i%5 == 0:
		meanBlock.append(x)
		if x <= 416:
			meanBlock2.append(x)
		else:
			meanBlock2.append(416)


i = 0
k = 0
sum12 = 0
sum13 = 0
sum14 = 0
sumBlock = 0
mean12 = []
mean13 = []
mean14 = []
for x in time13[0:]:
	sum12 += time12[i]
	sum13 += time13[i]
	sum14 += time14[i]
	i += 1
	if i%5 == 0:
		k += 1
		sum12 = sum12/5
		sum13 = sum13/5
		sum14 = sum14/5
		mean12.append(sum12)
		mean13.append(sum13)
		mean14.append(sum14)
		sum12 = 0
		sum13 = 0
		sum14 = 0

print mean12
print mean13
print mean14
print meanBlock
print meanBlock2

assiX = []
i = 4
for x in xrange(4,13):
	assiX.append(i)
	i+=1;

speedUp14_13 = []
speedUp14_12 = []
speedUp12_13 = []
efficency14_12 = []
efficency14_12_2 = []

for x in xrange(4,13):
	speedUp14_13.append(mean13[x-4]/mean14[x-4])
	speedUp14_12.append(mean12[x-4]/mean14[x-4])
	speedUp12_13.append(mean13[x-4]/mean12[x-4])
	efficency14_12.append(mean12[x-4]/(mean14[x-4]*meanBlock[x-4]))
	efficency14_12_2.append(mean12[x-4]/(mean14[x-4]*meanBlock2[x-4]))

plt.subplot(4,1,1)
plt.plot(assiX,mean12,linewidth=2.0,color='red',label="1block")
plt.plot(assiX,mean13,linewidth=2.0,color='blue',label="1blockSimple")
plt.plot(assiX,mean14,linewidth=2.0,color='green',label="Nblock")

block = mpatches.Patch(color='red', label='1block')
blockSimple = mpatches.Patch(color='blue', label='1blockSimple')
Nblock = mpatches.Patch(color='green', label='Nblock')

plt.legend(handles=[block, blockSimple, Nblock],loc=2,prop={'size': 8});
plt.title('compare time versions')
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(assiX,speedUp14_13,linewidth=2.0,color='red',label="NBlock vs 1BlockSimple")
plt.plot(assiX,speedUp14_12,linewidth=2.0,color='blue',label="NBlock vs 1Block")
plt.plot(assiX,speedUp12_13,linewidth=2.0,color='green',label="1Block vs 1BlockSimple")

block = mpatches.Patch(color='red', label='NBlock vs 1BlockSimple')
blockSimple = mpatches.Patch(color='blue', label='NBlock vs 1Block')
Nblock = mpatches.Patch(color='green', label='1Block vs 1BlockSimple')

plt.legend(handles=[block, blockSimple, Nblock],prop={'size': 8},loc=2);
plt.title('compare speedUps')
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(assiX,efficency14_12,linewidth=2.0,color='red')
plt.title('efficiency comparison with one block')
plt.grid(True)

plt.subplot(4,1,4)
plt.plot(assiX,efficency14_12_2,linewidth=2.0,color='red')
plt.title('efficiency comparison with one block with CUDAcores')
plt.grid(True)

plt.show()







