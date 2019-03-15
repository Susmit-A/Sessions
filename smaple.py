from sklearn.datasets import load_iris

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

import numpy

dataset = load_iris()

X = dataset['data']
Y = dataset['target']

print(numpy.shape(X))

x_train, x_test, y_train, y_test = train_test_split(X, Y)

print(numpy.shape(x_train))
print(numpy.shape(x_test))


model = MLPClassifier(activation='relu', alpha=0.01, max_iter=1000)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(predictions)

count = 0
for i in range(len(predictions)):
    if predictions[i]==y_test[i]:
        count += 1

print(count*100/len(predictions))
