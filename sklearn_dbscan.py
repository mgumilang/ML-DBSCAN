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
print (labels_sklearn)
