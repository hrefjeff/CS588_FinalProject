#!/usr/bin/env python3

import constants
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

from sklearn.metrics import davies_bouldin_score, silhouette_score

exercise = 1
unit = 2

# Load dataset for each subject
for subject in range(1, constants.SUBJECTS + 1):
    x = pd.read_csv(f's{subject}/e{exercise}/u{unit}/test.txt', delimiter=';')

    # Drop the time index as it's not a feature
    x = x.drop(columns=['time index'])

    # Data visualization
    plt.scatter(x['acc_x'], x['acc_y'], s=50)
    plt.xlabel('Accelerometer X', fontsize=14)
    plt.ylabel('Accelerometer Y', fontsize=14)
    plt.savefig(f'plots/scatterplot-s{subject}-e{exercise}-u{unit}.png')
    plt.figure()

# TODO: Run PCA to reduce dimensions down to 2 and justify it
# TODO: Run clustering algorithm k means on each subject's e1-u2

exit()


# Load Iris Data
iris = load_iris()

x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])

# Data visualization
colors = np.array(['red', 'green', 'blue'])
fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=50)
axes[1].scatter(x['Petal Length'], x['Petal Width'], c=colors[iris.target], s=50)
axes[0].set_xlabel('Sepal length', fontsize=14)
axes[0].set_ylabel('Sepal width', fontsize=14)
axes[1].set_xlabel('Petal length', fontsize=14)
axes[1].set_ylabel('Petal width', fontsize=14)

fig.savefig('iris-scatterplot.png')
plt.figure()

###### Step 1: Pre-clustering

# K-Means clustering
cost = []
for i in range(1,7):
    km = KMeans(n_clusters=i, max_iter=500)
    # Perform K-means clustering on data X
    km.fit(x)
    # Calculates squared error for clustered points
    cost.append(km.inertia_)

# Plot the cost against K values
plt.plot(range(1, 7), cost, color='b', linewidth='4')
plt.xlabel('Value of K')
plt.ylabel('Squared Error (Cost)')
plt.title('Elbow Method')
plt.savefig('elbow-method.png')
plt.figure()

# Elbow method explanation: the bend in the elbow tells you how many
# clusters are in the dataset. In the iris dataset above, we need either 2
# or 3 clusters. Not sure how many clusters though.

###### Step 2: Cluster analysis

# K-Means clustering
km = KMeans(n_clusters=3, random_state=111)

# Perform K-means clustering on data X
km.fit(x)

# Display centroids
centroids = pd.DataFrame(km.cluster_centers_, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
print(centroids)

# Predictions label colors
color1 = np.array(['green', 'red', 'blue'])
pred_y = pd.DataFrame(km.labels_, columns=['Target']) # Membership matrix

# Data visualization - before and after clustering for sepal length vs. sepal width
fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=50)
axes[1].scatter(x['Sepal Length'], x['Sepal Width'], c=color1[pred_y['Target']], s=50)
axes[1].scatter(centroids['Sepal Length'], centroids['Sepal Width'], c='k', s=70)
axes[0].set_xlabel('Sepal length', fontsize=14)
axes[0].set_ylabel('Sepal width', fontsize=14)
axes[0].set_title('Before K-Means clustering', fontsize=14)

axes[1].set_xlabel('Sepal length', fontsize=14)
axes[1].set_ylabel('Sepal width', fontsize=14)
axes[1].set_title('After K-Means clustering', fontsize=14)

fig.savefig('iris-clustering-SLSW.png')
plt.figure()

# Data visualization - before and after clustering for petal length vs. petal width
fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']], s=50)
axes[1].scatter(x['Petal Length'], x['Petal Width'], c=color1[pred_y['Target']], s=50)
axes[1].scatter(centroids['Petal Length'], centroids['Petal Width'], c='k', s=70)
axes[0].set_xlabel('Petal length', fontsize=14)
axes[0].set_ylabel('Petal width', fontsize=14)
axes[0].set_title('Before K-Means clustering', fontsize=14)

axes[1].set_xlabel('Petal length', fontsize=14)
axes[1].set_ylabel('Petal width', fontsize=14)
axes[1].set_title('After K-Means clustering', fontsize=14)

fig.savefig('iris-clustering-PLPW.png')

# What's the lesson from all this? If the data is well separated, it is very
# easy to get good results with k-means clustering classification.
# Since I have the ground truth, I can test the accuracy

###### Step 3: Cluster Validity

# DB Index (CHOOSE MINIMUM VALUE)
print(f'Davis Bouldin index for clutsers = 3 is {davies_bouldin_score(x, km.labels_)}')
# Davis Bouldin index for clutsers = 3 is 0.6660385791628493

# Silhouette index gives the average value for all the samples
print(f'Silhouette index for clusters = 3 is {silhouette_score(x, km.labels_)}\n')
# Silhouette index for clusters = 3 is 0.551191604619592

# To generate a range of cluster validity scores
num_clusters = [2, 3, 4, 5, 6]

# cluster validity over a range of clusters
for i in num_clusters:
    # Perform K-means
    km1 = KMeans(n_clusters=i)
    clabel = km1.fit_predict(x)

    # DB Index
    print(f'Davis Bouldin index for clutsers = {i} is {davies_bouldin_score(x, clabel)}')

    # Silhouette index gives the average value for all the samples
    print(f'Silhouette index for clusters = {i} is {silhouette_score(x, clabel)}')

# -------------------------------------------------------------------------|
# Davis Bouldin index for clutsers = 2 is 0.40429283717304343 | <--- min   |
# Silhouette index for clusters = 2 is 0.6810461692117462     | <--- max   |
# -------------------------------------------------------------------------|
# Davis Bouldin index for clutsers = 3 is 0.9937437429737788
# Silhouette index for clusters = 3 is 0.5185675688773279
# Davis Bouldin index for clutsers = 4 is 0.7757009440067065
# Silhouette index for clusters = 4 is 0.4974551890173751
# Davis Bouldin index for clutsers = 5 is 0.8129584871265321
# Silhouette index for clusters = 5 is 0.49394444148143263
# Davis Bouldin index for clutsers = 6 is 0.933282304641209
# Silhouette index for clusters = 6 is 0.35962925025285736

# So this means that our algorithms are seeing 2 clusters

###### Hierarchical clustering
import plotly as ff
from scipy.cluster import hierarchy

# Get the linkage matrix using dissmilarity - euclidean or similarity - cosine
linkage = hierarchy.linkage(x, metric='euclidean')

# Plot the dendrogram
fig = plt.figure(figsize=(25,10))
s = hierarchy.dendrogram(linkage, leaf_font_size=12)
fig.savefig('dendogram.png')

# Can be used for
# 1. Simple identification of patterns
# 2. Feature selection
# 3. Data rebalancing
# 4. Pruning outliers
