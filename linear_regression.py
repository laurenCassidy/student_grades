import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read student performance data from a CSV file using Pandas
data = pd.read_csv("student-mat.csv", sep=";")

# Select specific columns ('G2' and 'G3') as our features
data = data[["G2", "G3"]]

# Define the target variable (the variable we want to predict)
predict = "G3"

# Separate the features (X) and the target variable (y)
X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# train model:
"""
# Create a linear regression model
linear = linear_model.LinearRegression()

# Train the model on the training data
linear.fit(x_train, y_train)

# Evaluate the model's accuracy on the test data
acc = linear.score(x_test, y_test)

# Print the accuracy
print(acc)

# Save the trained model to a file
with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)
"""

# Load a pre-trained model from a pickle file
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Print the coefficients and intercept of the linear model
print(linear.coef_)      # Coefficients for features
print(linear.intercept_) # Intercept

# Create a scatter plot using Matplotlib
p = 'G1'  # Variable 'p' to represent another feature
style.use("ggplot")  # Set the style for the plot
pyplot.scatter(data[p], data['G3'])  # Plot a scatter graph
