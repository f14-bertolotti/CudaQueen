from matplotlib import pyplot
from pylab import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

QueenSet = 10

maxDiff = 0

sampleNumber = 5

matWQSet = genfromtxt("outWQ10")
matNQSet = genfromtxt("outNQ10")

matLKWQ = np.zeros((QueenSet,QueenSet))
matLKNQ = np.zeros((QueenSet,QueenSet))

for x in matWQSet:
	matLKWQ[int(x[7])][int(x[8])] += x[9]

for x in matNQSet:
	matLKNQ[int(x[7])][int(x[8])] += x[9]

for i in range(QueenSet):
	for j in range(QueenSet):
		matLKWQ[i][j] = (matLKWQ[i][j] / sampleNumber)/1000
		matLKNQ[i][j] = (matLKNQ[i][j] / sampleNumber)/1000

X = range(QueenSet)
Y = range(QueenSet)
X, Y = np.meshgrid(X, Y)

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
hf.set_size_inches(16,12)
ha.plot_wireframe(X,Y,matLKWQ,color='red')
ha.plot_wireframe(X,Y,matLKNQ,color='blue')
plt.title(str(QueenSet)+' queens')
WQueue = mpatches.Patch(color='red', label='with queue')
NQueue = mpatches.Patch(color='blue', label='withouth queue')
ha.legend(handles=[WQueue, NQueue]);
ha.set_xlabel('level discriminant 1');
ha.set_ylabel('level discriminant 2');
ha.set_zlabel('time (s)');
plt.tight_layout()
plt.show()
