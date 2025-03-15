import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

x_train = pd.read_csv('data_train.csv')
y_train = pd.read_csv('ans_train.csv')
x_test = pd.read_csv('data_test.csv')
y_test = pd.read_csv('ans_test.csv')

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

predict = clf.predict(x_test)

precision = precision_score(y_test, predict, average="macro")
recall = recall_score(y_test, predict, average="macro")
f1 = f1_score(y_test, predict, average="macro")
accuracy = accuracy_score(y_test, predict)

conf = confusion_matrix(y_test, predict)

print("SVM Poly\n")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1)
print("Confusion Matrix:\n", conf)