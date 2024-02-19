import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np

import os

current_directory = os.getcwd()

file_name = r'data\2D_randgen.csv'

full_path = os.path.join(current_directory, file_name)

try:
    df = pd.read_csv(file_name, header=0, index_col=0, )
except FileNotFoundError:
    raise Exception(f"file path {full_path} not found")


r = 1   # Radius

# Convert cylindrical coordinates to Cartesian coordinates
df['x'] = r * np.cos(df['theta'])
df['y'] = r * np.sin(df['theta'])


# Data Standardization

scaler = StandardScaler()

df[['x_t', 'y_t', 'z_t']] = scaler.fit_transform(df[['x', 'y', 'z']])

def optimum_k_means(data, max_k):
    means = []
    inertias = []
    sil = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

        print(f"{k}: {(inertias[k-1] - inertias[k-2])}")

    # Calculate Silhouette Scores

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters = k).fit(data)
        labels = kmeans.labels_
        sil.append(silhouette_score(data, labels, metric = 'euclidean'))

    print("\nSilhouette Score")
    print(sil)
    max_sil_k = sil.index(max(sil)) + 1
    print(f"{max_sil_k} : {max(sil)}")

    #Generate elbow plot
    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

optimum_k_means(df[['x_t', 'y_t', 'z_t']], 10)
    
kmeans = KMeans(n_clusters=7)
kmeans.fit(df[['x_t', 'y_t', 'z_t']])

df['kmeans_7'] = kmeans.labels_

print(df)

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(df['x'], df['y'], df['z'], c = df['kmeans_7'], s = 50)
ax.set_title('CMS Clustering')

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

plt.show()