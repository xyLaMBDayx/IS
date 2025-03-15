import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

x_train = pd.read_csv('data_train.csv')
y_train = pd.read_csv('ans_train.csv')
x_test = pd.read_csv('data_test.csv')
y_test = pd.read_csv('ans_test.csv')

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

from sklearn import svm, datasets

poly = svm.SVC(kernel='poly', degree=5, C=10,probability = True)
poly.fit(x_train, y_train)

svmpredictionPoly = poly.predict(x_test)

precision_svmPoly = precision_score(y_test, svmpredictionPoly, average="macro")
recall_svmPoly = recall_score(y_test, svmpredictionPoly, average="macro")
f1_svmPoly = f1_score(y_test, svmpredictionPoly, average="macro")
accuracy_svmPoly = accuracy_score(y_test, svmpredictionPoly)

conf_matrix_svmPoly = confusion_matrix(y_test, svmpredictionPoly)

print("SVM Poly\n")
print("Accuracy: ", accuracy_svmPoly)
print("Precision: ", precision_svmPoly)
print("Recall: ", recall_svmPoly)
print("F1-Score: ", f1_svmPoly)
print("Confusion Matrix:\n", conf_matrix_svmPoly)