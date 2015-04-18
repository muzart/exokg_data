import numpy as np
dataset = np.loadtxt(fname='exokg.data',delimiter=';')
col_count = 10 #number of columns
X = dataset[:,0:col_count]
y = dataset[:,col_count]
np.random.seed(0)
indices = np.random.permutation(len(X))
k = (int) (-len(X)*0.15)
X_train = X[indices[:k]]
y_train = y[indices[:k]]
X_test = X[indices[k:]]
y_test = y[indices[k:]]

def calculateData(expected,predicted):
    length = len(expected)
    count = 0
    for i in range(length):
        if expected[i] == predicted[i]:
            count += 1
    print("Total = ",length)
    print("Equals = ",count,count/length*100,"%")
    print("Not Equals = ",length-count,(length-count)/length*100,"%")
    print("------------------------------------------------------------------------")

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
calculateData(expected,predicted)

# Logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0)
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
calculateData(expected,predicted)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
calculateData(expected,predicted)

# K blijaywi sosed
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
calculateData(expected,predicted)

# Metod opornix vektor
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
calculateData(expected,predicted)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
calculateData(expected,predicted)

#  Perceptron
from sklearn.linear_model import Perceptron
model = Perceptron()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
calculateData(expected,predicted)
