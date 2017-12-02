"""
    Pseudocode from wikipedia
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn import metrics

eps = 2
minpts = 2
NOISE = -1
UNDEFINED = -999

# Create condensed distance vector    
df = pd.read_csv('normalized_undersampled.csv')
dis_condensed = pdist(df, 'euclidean')
N = len(df)

# Condensed Index
def square_to_condensed(i, j, n):
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    res = n*j - j*(j+1)/2 + i - 1 - j
    return int(res)

# Range Query function
def RangeQuery(q_id, eps):
    """
        Params
            eps : epsilon
            q_id : id of dataframe object to be queried
    """
    neighbors = []
    for j in range(N):
        if q_id == j:
            neighbors.append(j)
            continue
#        print ('{}, {}, {}, {}\n'.format(q_id,j,N, dis_condensed[square_to_condensed(q_id, j, N)]))
        if dis_condensed[square_to_condensed(q_id, j, N)] <= eps:
            neighbors.append(j)
    return neighbors

# Init global labels / cluster
# Cluster started from 1 to N
# Noise defined as 0.0
# Undefined defined as -1.0
labels = []
for x in range(N):
    labels.append(UNDEFINED)

# DBScan function
import timeit
def DBScan(eps, minpts):
    """
        Params
            eps : epsilon
            minpts : mininum pts
    """
    cluster = 0 # initial cluster
    for i in range(N):
        start = timeit.default_timer()
        if labels[i] != UNDEFINED:
            continue
        neighbors = RangeQuery(i, eps)
        if len(neighbors) < minpts:
            labels[i] = NOISE #label as noise
            continue
        cluster += 1
        labels[i] = cluster
        neighbors = {x for x in neighbors if x != i} #remove current record in neighbors list
        while neighbors:
            neighbor = neighbors.pop()
            if labels[neighbor] == NOISE:
                labels[neighbor] = cluster
            if labels[neighbor] != UNDEFINED:
                continue
            labels[neighbor] = cluster
            n_neighbors = set(RangeQuery(neighbor, eps))
            if len(n_neighbors) >= minpts:
                neighbors = set().union(neighbors, n_neighbors)
        stop = timeit.default_timer()
        print ('Time elapsed for {}/{} loop : {}'.format(i, N, stop-start))

# Do DBScan
print ("Do DBScan ....")
DBScan(eps, minpts)

# Save labels
#with open ('labels.txt', 'w') as f:
#    for _id, label in enumerate(labels):
#        f.write('{},{}\n'.format(_id, label))

# Convert label
for _id, label in enumerate(labels):
    if label == NOISE:
        continue
    else:
        labels[_id] = labels[_id] - 1
              
# From labels to clusters
clusters = []
noise = []
for _id, label in enumerate(labels):
    if label not in clusters and label != NOISE and label != UNDEFINED:
        clusters.append(label)
    if label == NOISE:
        noise.append(_id)
        
# Create per clusters membership
cluster_member = []
for cluster in clusters:
    members = []
    for _id, label in enumerate(labels):
        if label == cluster:
            members.append(_id)
    cluster_member.append(members)

# Print result
print ()

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

# Compute cooccurence matrix
cooccurrence_matrix = []
for member in cluster_member:
    conf_mat = []
    classified_0 = 0.0
    classified_1 = 0.0
    for el in member:
        if real_label[el] == 1.0:
            classified_1 += 1
        else:
            classified_0 += 1
    conf_mat.append(classified_0)
    conf_mat.append(classified_1)
    cooccurrence_matrix.append(conf_mat)

print (cooccurrence_matrix)

# Compute purity from cooccurence matrix
total_data = 0
total_single_class = 0
for cluster_members in cooccurrence_matrix:
    total_single_class += max(cluster_members)
    for member in cluster_members:
        total_data += member
purity = (total_single_class/total_data)*100
    
print ('Purity : {}%'.format(purity))

# RI
from scipy.misc import comb

# There is a comb function for Python which does 'n choose k'                                                                                            
# only you can't apply it to an array right away                                                                                                         
# So here we vectorize it...                                                                                                                             
def myComb(a,b):
  return comb(a,b,exact=True)

vComb = np.vectorize(myComb)

def get_tp_fp_tn_fn(cooccurrence_matrix):
  tp_plus_fp = vComb(cooccurrence_matrix.sum(0, dtype=int),2).sum()
  tp_plus_fn = vComb(cooccurrence_matrix.sum(1, dtype=int),2).sum()
  tp = vComb(cooccurrence_matrix.astype(int), 2).sum()
  fp = tp_plus_fp - tp
  fn = tp_plus_fn - tp
  tn = comb(cooccurrence_matrix.sum(), 2) - tp - fp - fn

  return [tp, fp, tn, fn]

cooccurrence_matrix = np.array(cooccurrence_matrix)

# Get the stats                                                                                                                                        
tp, fp, tn, fn = get_tp_fp_tn_fn(cooccurrence_matrix)

print ("TP: %d, FP: %d, TN: %d, FN: %d" % (tp, fp, tn, fn))
# Print the measures:                                                                                                                                  
print ("Rand index: %f" % (float(tp + tn) / (tp + fp + fn + tn)))

precision = float(tp) / (tp + fp)
recall = float(tp) / (tp + fn)

print ("Precision : %f" % precision)
print ("Recall    : %f" % recall)
print ("F1        : %f" % ((2.0 * precision * recall) / (precision + recall)))


