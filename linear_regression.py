import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G2", "G3"]]

predict = "G3"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

"""linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)"""

pickle_in = open("studentmodel.pickle", "rb")

linear  = pickle.load(pickle_in)

print(linear.coef_)
print(linear.intercept_)

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data['G3'])
