import matplotlib.pyplot as plt
import numpy as np

from Generate_Dataset import generate_data, plot_data

from Implement_logisticRegression import train,predict
from Visualize_Decision_BBoundary import plot_decision_boundary

X,y = generate_data()
plot_data(X,y)
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = predict(np.c_[xx.ravel(), yy.ravel()], weights,bias )
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap="bwr", alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr")
plt.title("Logistic Regression Decision Boundary")
plt.show()