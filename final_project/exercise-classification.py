#!/usr/bin/env python3

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

subject = 1
exercise = 1
unit = 2

def plot_per_class_accuracy(classifier, X, y, label, feature_selection = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=101)
    pipeline = Pipeline([("scalar", MinMaxScaler()), ("classifier", classifier)])
    pipeline.fit(X_train, y_train)
    disp = plot_confusion_matrix(pipeline, X_test, y_test, cmap=plt.cm.Blues)
    plt.title(label)
    plt.savefig(f'plots/confusionmatrix/cm-{label}.png')
    true_positive = disp.confusion_matrix[1][1]
    false_negative = disp.confusion_matrix[1][0]
    print(label + " - Sensitivity: ", true_positive/(true_positive+false_negative))
    print()

# Load dataset
infile = f's{subject}/e{exercise}/u{unit}/test-labeled.csv'
df = pd.read_csv(infile, delimiter=';')
unmodified_df = pd.read_csv(infile, delimiter=';')

# Drop the time index as it's not a feature
df = df.drop(columns=['time index'])

# Iris data statistics
# Trying to use linear regression to predict any of
# the properties of the iris
#print(iris_df.describe())

#        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# count         150.000000        150.000000         150.000000        150.000000
# mean            5.843333          3.057333           3.758000          1.199333
# std             0.828066          0.435866           1.765298          0.762238
# min             4.300000          2.000000           1.000000          0.100000
# 25%             5.100000          2.800000           1.600000          0.300000
# 50%             5.800000          3.000000           4.350000          1.300000
# 75%             6.400000          3.300000           5.100000          1.800000
# max             7.900000          4.400000           6.900000          2.500000

# Looking at the table above, the goal is to predict things. So drop
# one column like petal width, and based on the other 3 columns,
# predict what the petal width would be. Drop the species, and use all
# columns to predict the species

# Variables (Dropping petal length to show we can predict it)
# X = df.drop(labels='acc_y', axis=1)
# y = df['acc_y']
X = df.drop(labels='label', axis=1)
y = df['label']

# Splitting the Dataset
# In the real world, we test only a small sample
# Use minimum amount of samples to pridict vast majority
# 10% train, 90% test. 20% train, 80% test. 50% train, 50% test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=111) # train 20%
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111) # train 80%

# LESS CONFUSING WAY
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=111)
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=111)

# Instantiate all classifiers
lr = LinearRegression()
knn_classifier = KNeighborsClassifier(n_neighbors=3)
nb_classifier = GaussianNB()

# Fit Models
lr.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
nb_classifier.fit(X_train, y_train)

# LR Prediction
y_pred = lr.predict(X_test)

# LR coefficients - beta/slope
print('LR beta/slope coefficient: ', lr.coef_)

# LR coefficients - alpha/slope intercept
print('LR alpha/slope intercept coefficient: ', lr.intercept_)

# coefficient of dtermination: 1 is the perfect prediction
print('Coefficient of determination: ', r2_score(y_test, y_pred))

# Model performance - Error (Quantitative analysis)
print('Root Mean Squared Error (RMSE): ', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean Squared Error (MSE): ', mean_squared_error(y_test, y_pred))

# KNN Prediction
y_pred = knn_classifier.predict(X_test)

# Evaluate the classifier
print('\n')
print('KNN')
print(classification_report(y_test, y_pred))


# Predict on the test data
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier
print('\n')
print('Naive Bayes')
print(classification_report(y_test, y_pred))

# LR beta/slope coefficient:  [ 0.61207486  0.78013546 -0.35124986 -0.34942695]
# The 4 values are because there are 4 different attributes

# LR alpha/slope intercept coefficient:  1.8102535296995388
# This is the prediction for sepal length, when everything is 0.
# It'll start at the length 1.8cm

# Coefficient of determination:  0.8779831575882425
# Root Mean Squared Error (RMSE):  0.2934103969442157
# This is the error. It is a little high. Generally want this to be really low

# Mean Squared Error (MSE):  0.08608966103496224
# We want to find absolute value, don't care positive or negative

# Try adjusting test_size = 90%

# LR beta/slope coefficient:  [ 1.19265148  1.08254024 -1.87795011  0.63919001]
# LR alpha/slope intercept coefficient:  -0.1801215992495555
# Coefficient of determination:  0.6931549115115272
# Root Mean Squared Error (RMSE):  0.4558346450891425 <--- this increased in value
# Mean Squared Error (MSE):  0.20778522366354452      <--- this increased in value

# RMSE went up 0.29 --> 0.45 when test was 90%
# THIS IS CALCULATING WITHIN .45 CM OF ERROR
# Just trained the model on 10% of the data, so it's not very accurate
# Lower samples used to train model, the faster it is to train

classifier_labels = {
    "Random Forest": (RandomForestClassifier(random_state=1), "red"),
    "kNN": (knn_classifier, "blue"),
    "Guassian Naive Bayes": (nb_classifier, "lime")
}

for label in classifier_labels:
    classifier = classifier_labels[label][0]
    plot_per_class_accuracy(classifier, X, y, label)

# MOVING ON TO PREDICTION

# # Predicting a new data point
# actual_df = unmodified_df.loc[5043]

# # Create a new dataframe
# d = {
#     'acc_x' : [actual_df['acc_x']],
#     'acc_y' : [actual_df['acc_y']],
#     'acc_z' : [actual_df['acc_z']],
#     'gyr_x' : [actual_df['gyr_x']],
#     'gyr_y' : [actual_df['gyr_y']],
#     'gyr_z' : [actual_df['gyr_z']],
#     'mag_x' : [actual_df['mag_x']],
#     'mag_y' : [actual_df['mag_y']],
#     'mag_z' : [actual_df['mag_z']],
#     'label' : [actual_df['label']]
# }

# test_df = pd.DataFrame(data=d)

# # print(test_df)

# X_test = test_df.drop('label', axis=1)
# y_test = test_df['label']

# # Predict the new data point using LR
# pred = lr.predict(X_test)

# print('Predicted label: ', pred[0])
# print('Actual label: ', actual_df['label'])
