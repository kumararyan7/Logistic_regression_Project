from sklearn.datasets import make_classification

import numpy as np 
import matplotlib.pyplot as plt 

X,y = make_classification(n_samples=200,n_features=2,n_classes=2,n_informative=2,n_redundant=0,random_state=42)

plt.scatter(X[:,0],X[:,1],c=y,cmap="bwr")
plt.title("Toy Dataset")
plt.show()