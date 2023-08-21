
#Importing required modules

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#Load Data
data = load_digits().data
pca = PCA(1)

#Transform the data
df = pca.fit_transform(data)

df.shape

#Import required module
from sklearn.cluster import KMeans

#Initialize the class object
kmeans = KMeans(n_clusters=3)

#predict the labels of clusters.
label = kmeans.fit_predict(df)

print(label)

u_labels = np.unique(label)

#plotting the results:

for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()