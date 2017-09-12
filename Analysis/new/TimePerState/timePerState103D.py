from matplotlib import pyplot
from pylab import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

QueenSet = 10
sampleNumber = 5



timesPerStateLK = np.zeros((QueenSet,QueenSet))

for l in range(QueenSet-1):
	for k in range(QueenSet-1):
		if l < k:
		 	matSet = genfromtxt("out"+str(QueenSet)+"-"+str(l)+"-"+str(k))
		 	for x in matSet:
		 		timesPerStateLK[int(x[7])][int(x[8])] += x[9]/x[2];
		 		
for i in range(QueenSet):
	for j in range(QueenSet):
		timesPerStateLK[i][j] = (timesPerStateLK[i][j] / sampleNumber)

for i in range(QueenSet):
	timesPerStateLK[i][9] = np.nan

X = range(QueenSet)
Y = range(QueenSet)
X, Y = np.meshgrid(X, Y)

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
hf.set_size_inches(16,12)
ha.plot_wireframe(X,Y,timesPerStateLK,color='black')
plt.title(str(QueenSet)+' queens')
ha.set_xlabel('level discriminant 2');
ha.set_ylabel('level discriminant 1');
ha.set_zlabel('time (s)');
plt.tight_layout()
plt.show()

