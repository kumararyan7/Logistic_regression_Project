from Generate_Dataset import generate_data, plot_data
from Implement_logisticRegression import train,predict
from Visualize_Decision_BBoundary import plot_decision_boundary
import numpy as np

X,y = generate_data()
plot_data(X,y)

weights,bias = train(X,y)
y_pred = predict(X,weights,bias)
accuracy = np.mean(y_pred == y)
print("Training Accuracy:",accuracy)

plot_decision_boundary(X,y,weights,bias)