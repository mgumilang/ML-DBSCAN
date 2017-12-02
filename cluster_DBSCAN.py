#%%
import pandas as pd
import numpy as np
import math

eps = 2
minpts = 2

#%%
df = pd.read_csv('normalized_10.csv')

#%%
dis_table = np.zeros([df.loc[:, '0'].count(), df.loc[:, '0'].count()])

for x in range(df.loc[:, '0'].count()):
    for y in range(x+1, df.loc[:, '0'].count()):
        print(x, y)
        a = df.loc[x].values.tolist()
        b = df.loc[y].values.tolist()
        sum = 0
        for i in range(len(a)):
            sum += (a[i] - b[i]) ** 2
        dis_table[x][y] = math.sqrt(sum)    
        
#%%
with open ('dis_table.txt', 'w')  as f:
    for x in range(df.loc[:, '0'].count()):
        for y in range(x, df.loc[:, '0'].count()):
            f.write('{} {} {}\n'.format(x, y, dis_table[x][y]))
            
#%%
eps_list = []
for x in range(df.loc[:, '0'].count()):
    x_list = []
    for y in range(x):
        if (dis_table[y][x] < eps):
            x_list.append(y)
    for y in range(x+1, df.loc[:, '0'].count()):
        if (dis_table[y][x] < eps):
            x_list.append(y)
    eps_list.append(x_list)
    
#%%
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