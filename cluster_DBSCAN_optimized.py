"""
    Pseudocode from wikipedia
"""
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

total_data = 0
total_single_class = 0
confusion_matrix = []
for member in cluster_member:
    conf_mat = []
    classified_0 = 0.0
    classified_1 = 0.0
    for el in member:
        if real_label[el] == 1.0:
            classified_1 += 1
        else:
            classified_0 += 1
        total_data += 1
    conf_mat.append(classified_0)
    conf_mat.append(classified_1)
    confusion_matrix.append(conf_mat)
    total_single_class += max(classified_0, classified_1)

print (labels)
print ('Purity : {}%'.format((total_single_class/total_data)*100))
print('Estimated number of clusters: %d' % len(clusters))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(real_label, labels))
print("Completeness: %0.3f" % metrics.completeness_score(real_label, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(real_label, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(real_label, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(real_label, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df, labels))


    