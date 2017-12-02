import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

eps = 2
minpts = 2

df = pd.read_csv('normalized_train.csv')

N = df.loc[:, '0'].count()
eps_table = np.zeros([df.loc[:, '0'].count(), df.loc[:, '0'].count()]) #init table

print ('Creating Eps List')
eps_list = []
for x in range(N):
    print (x)
    x_list = []
    for y in range(N):
        if (y == x):
            continue
        else:
            if (eps_table[x][y] == 2.0 or eps_table[y][x] == 2.0):
                continue
            elif(eps_table[x][y] == 1.0 or eps_table[y][x] == 1.0):
                x_list.append(y)
            else:
                rc1 = df.iloc[x, :]
                rc2 = df.iloc[y, :]
                s = abs(euclidean(rc1, rc2))
                if (s < eps):
                    eps_table[x][y] = 1.0 #marked as s < epsilon
                    eps_table[y][x] = 1.0
                    x_list.append(y)
                else:
                    eps_table[x][y] = 2.0 #marked as to far (s>=epsilon)
                    eps_table[y][x] = 2.0
    eps_list.append(x_list)

print ('Eps List Created')    
core_list = []
for e in range(len(eps_list)):
    if (len(eps_list[e])+1 >= minpts):
        core_list.append(e)
            
noise = []
border = []
cluster = []

def DBSCAN(e, z):
    try:
        if e in cluster[z]:
            return
    except:
        cluster.append([])
    if e in noise:
        return
    if (e not in core_list):
        isborder = False
        for point in eps_list[e]:
            if point in core_list:
                border.append(e)
                cluster[z].append(e)
                isborder = True
        if not isborder:
            noise.append(e)
        for point in eps_list[e]:
            DBSCAN(point, z)
    else:
        cluster[z].append(e)
        for point in eps_list[e]:
            DBSCAN(point, z)

z = 0
for e in range(len(eps_list)):
    found = False
    for clust in cluster:
        if e in clust:
            found = True
            break
    if found:
        continue
    if e in noise:
        continue
    DBSCAN(e, z)
    if e not in noise:
        z += 1