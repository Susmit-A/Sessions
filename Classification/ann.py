from sklearn.neural_network import MLPClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def check_accuracy(prediction, actual):
    count = 0
    for i in range(len(prediction)):
        if prediction[i] == actual[i]:
            count += 1
    return (count/len(prediction))*100


iris_data = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target)

for i in range(10):
    model = MLPClassifier(activation='relu', alpha=0.01, max_iter=1000)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    print(check_accuracy(pred, y_test))
    del model
