
#pca on Iris dataset
import pandas as pd
data = pd.read_csv(filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None,sep=',')
#loading the dataset
X = data.ix[:,0:3].values
y = data.ix[:,4].values

import numpy as np

M = np.mean(X, axis=0)
X=X-M
X_trans = np.matrix.transpose(X)

#finding the covariance matrix

cov_mat = (X.dot(X_trans))/150

#finding the eigen value
  
u,s,v = np.linalg.svd(cov_mat)
u_reduce = u[:,0:1]
print np.shape(X)
#get the reduced matrix values
z=np.matrix.transpose(u_reduce).dot(X)
print np.shape(z)
print z
print '1-PCA 1 dimesional PCA'
print u_reduce
print '2-PCA 2 dimensional PCA'
u_reduce = u[:,0:2]
print u_reduce

y_new = [] 
a=0
b=0
c=0
for i in range(0,len(y)):
	if y[i]=='Iris-setosa':
		y_new = y_new+ ['r']
		a=i
	elif y[i] == 'Iris-versicolor':
		y_new = y_new + ['g']
		b=i
	elif y[i] == 'Iris-virginica':
		y_new = y_new + ['b']
		c=i
print a
print b
print c
print np.shape(y)
print len(y_new)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

plt.hold(True)
print np.shape(u_reduce[0:49,0])
plt.figure(1)
plt.title('2-PCA represenration')
plt.plot(u_reduce[0:49,0],u_reduce[0:49,1],'r^',label='Iris-setosa')
plt.plot(u_reduce[50:99,0],u_reduce[50:99,1],'b^',label = 'Iris-versicolor')
plt.plot(u_reduce[100:150,0],u_reduce[100:150,1],'g^',label = 'Iris-virginica')
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)

plt.figure(2)
plt.title('plot for the combination of PCA and t-sne')
plt.hold(True)
plt.plot(u_reduce[0:49,0],u_reduce[0:49,1],'r^',label='Iris-setosa')
plt.plot(u_reduce[50:99,0],u_reduce[50:99,1],'b^',label = 'Iris-versicolor')
plt.plot(u_reduce[100:150,0],u_reduce[100:150,1],'g^',label = 'Iris-virginica')
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)

from sklearn.manifold import TSNE
u_reduce1 = TSNE(n_components=2).fit_transform(X)
plt.plot(u_reduce1[0:49,0],u_reduce1[0:49,1],'k^',label='sIris-setosa')
plt.plot(u_reduce1[50:99,0],u_reduce1[50:99,1],'m^',label = 'sIris-versicolor')
plt.plot(u_reduce1[100:150,0],u_reduce1[100:150,1],'y^',label = 'sIris-virginica')
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)



from sklearn.datasets import make_swiss_roll

data = make_swiss_roll(n_samples=150, noise=0.0, random_state=None)

X = data[0]
y = data[1]
M = np.mean(X, axis=0)
print np.shape(X)
X=X-M
print np.shape(np.matrix.transpose(X))
X_trans = np.matrix.transpose(X)
cov_mat = (X.dot(X_trans))/150
print cov_mat
u,s,v = np.linalg.svd(cov_mat)
print np.shape(u)
u_reduce = u[:,0:1]
print np.shape(u_reduce)
print np.shape(X)
u_reduce = u[:,0:2]
from sklearn.manifold import TSNE
u_reduce1 = TSNE(n_components=2).fit_transform(X)

plt.figure(3)
plt.title('comparision of PCA and T-sNe')
plt.plot(u_reduce[:,0],u_reduce[:,1],'r^',label='PCA')
plt.plot(u_reduce1[:,0],u_reduce1[:,1],'g^',label='t-sne')
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)

plt.show()


