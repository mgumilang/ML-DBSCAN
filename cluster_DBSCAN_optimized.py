"""
    Pseudocode from wikipedia
"""
import pandas as pd
from scipy.spatial.distance import pdist

eps = 2
minpts = 2

# Create condensed distance vector    
df = pd.read_csv('normalized_train.csv')
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
            continue
#        print ('{}, {}, {}, {}\n'.format(q_id,j,N, square_to_condensed(q_id,j,N)))
        if dis_condensed[square_to_condensed(q_id, j, N)] < eps:
            neighbors.append(j)
    return neighbors

# Init global labels / cluster
# Cluster started from 1 to N
# Noise defined as 0.0
# Undefined defined as -1.0
labels = []
for x in range(N):
    labels.append(-1.0)

# DBScan function
import timeit

def DBScan(eps, minpts):
    """
        Params
            eps : epsilon
            minpts : mininum pts
    """
    cluster = 1.0 # initial cluster
    for i in range(N):
        start = timeit.default_timer()
        
        if labels[i] != -1.0:
            continue
        neighbors = RangeQuery(i, eps)
        if len(neighbors) < minpts:
            labels[i] = 0.0 #label as noise
            continue
        cluster += 1.0
        labels[i] = cluster
        neighbors = [x for x in neighbors if x != i] #remove current record in neighbors list
        for neighbor in neighbors:
            if labels[neighbor] == 0.0:
                labels[neighbor] = cluster
            if labels[neighbor] == -1.0:
                continue
            labels[neighbor] = cluster
            n_neighbors = RangeQuery(neighbor, eps)
            if len(n_neighbors) >= minpts:
                neighbors = list(set().union(neighbors, n_neighbors))
        #Your statements here

        stop = timeit.default_timer()
        print ('Time elapsed for {}/{} loop : {}'.format(i, N, stop-start))

# Do DBScan
print ("Do DBScan ....")
DBScan(eps, minpts)
        
        
        
    