import pandas as pd,matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

x=pd.read_csv("data.csv").values
k_means=KMeans(n_clusters=2,random_state=0).fit(x)
gmm=GaussianMixture(n_components=2,random_state=0).fit(x)

k_label=k_means.labels_
gmm_label=gmm.predict(x)

print("k labels",k_label)
print("gmm label",gmm_label)
print(" k means silhouette score",silhouette_score(x,k_label))
print(" gmm silhouette score",silhouette_score(x,gmm_label))

plt.subplot(1,2,1);plt.title("kmeans");plt.scatter(x[:,0],x[:,1],c=k_label)
plt.subplot(1,2,2);plt.title("gmm");plt.scatter(x[:,0],x[:,1],c=gmm_label)
plt.tight_layout()
plt.show()
