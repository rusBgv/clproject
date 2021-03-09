import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import patches

import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

from scipy.stats import multivariate_normal

import warnings
warnings.filterwarnings('ignore')
N = 300
np.random.seed(3)
m1, cov1 = [9, 8], [[.5, 1], [.25, 1]] 
data1 = np.random.multivariate_normal(m1, cov1, N)
label1 = np.ones(N)

m2, cov2 = [6, 13], [[.5, -.5], [-.5, .1]] 
data2 = np.random.multivariate_normal(m2, cov2, N)
label2 = np.ones(N) * 2    

m3, cov3 = [4, 7], [[0.25, 0.5], [-0.1, 0.5]] 
data3 = np.random.multivariate_normal(m3, cov3, N)
label3 = np.ones(N) * 3

X = np.vstack((data1,np.vstack((data2,data3))))
y = np.concatenate((label1,label2,label3))

plt.scatter(X[:,0],X[:,1], c = y, cmap = 'viridis')
plt.xlabel('X1'), plt.ylabel('X2')
plt.show()