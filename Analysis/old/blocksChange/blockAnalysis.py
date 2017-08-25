from matplotlib import pyplot
from pylab import genfromtxt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

mat10 = genfromtxt("res10")
mat11 = genfromtxt("res11")
mat12 = genfromtxt("res12")

time10 = mat10[1:,9]
time11 = mat11[1:,9]
time12 = mat12[1:,9]

block10 = mat10[1:,3]
block11 = mat11[1:,3]
block12 = mat12[1:,3]

speedUp10 = []
speedUp11 = []
speedUp12 = []
for i in xrange(1,10):
	speedUp10.append(time10[i-1]/time10[i])
	speedUp11.append(time11[i-1]/time11[i])
	speedUp12.append(time12[i-1]/time12[i])

print 'speed up mean for nqueen=10 : '+str(np.mean(speedUp10))
print 'speed up mean for nqueen=11 : '+str(np.mean(speedUp11))
print 'speed up mean for nqueen=12 : '+str(np.mean(speedUp12))

plt.subplot(3,1,1)
plt.plot(block10,time10,linewidth=1.5,color='red')
plt.grid(True)
plt.title('nQueen = 10')

plt.subplot(3,1,2)
plt.plot(block11,time11,linewidth=1.5,color='green')
plt.grid(True)
plt.title('nQueen = 11')

plt.subplot(3,1,3)
plt.plot(block12,time12,linewidth=1.5,color='blue')
plt.grid(True)
plt.title('nQueen = 12')

plt.show()
