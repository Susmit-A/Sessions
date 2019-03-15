# To get an interactive graph, you may need to run this file from command line

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def check_accuracy(prediction, actual):
    count = 0
    for i in range(len(prediction)):
        if prediction[i] == actual[i]:
            count += 1
    return (count/len(prediction))*100


iris_data = load_iris()

print(np.shape(iris_data.data))
print(np.shape(iris_data.target))

x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target)

fig = plt.figure()
ax = Axes3D(fig)

x, y, z, w = np.transpose(x_train[:50, :4])
ax.scatter(x, y, z, c='r')

x, y, z, w = np.transpose(x_train[50:100, :4])
ax.scatter(x, y, z, c='g')

x, y, z, w = np.transpose(x_train[100: 150, :4])
ax.scatter(x, y, z, c='b')

plt.show()

for i in range(10):
    classifier = naive_bayes.GaussianNB()
    classifier.fit(x_train[:, :3], y_train)
    preds = classifier.predict(x_test[:, :3])
    print(check_accuracy(preds, y_test))

    del classifier
