import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics


eps = 2
minpts = 2

# Create condensed distance vector    
df = pd.read_csv('normalized_undersampled.csv')
dbs = DBSCAN(eps=eps, min_samples=minpts)
db = dbs.fit(df)
print (db)
labels_sklearn = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels_sklearn)) - (1 if -1 in labels_sklearn else 0)

# Evaluate
real_label = []
with open ('CencusIncomeUndersampled.csv', 'r') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
        x = line.split(',')
        if (x[-1] == '<=50K\n'):
            real_label.append(0.0)
        elif (x[-1] == '>50K\n'):
            real_label.append(1.0)
            
print (labels_sklearn)
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(real_label, labels_sklearn))
print("Completeness: %0.3f" % metrics.completeness_score(real_label, labels_sklearn))
print("V-measure: %0.3f" % metrics.v_measure_score(real_label, labels_sklearn))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(real_label, labels_sklearn))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(real_label, labels_sklearn))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df, labels_sklearn))
