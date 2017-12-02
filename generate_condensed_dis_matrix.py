# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 10:08:36 2017

@author: Ali-pc
"""
        
import pandas as pd
from scipy.spatial.distance import pdist
    
df = pd.read_csv('normalized_train.csv')

dis_condensed = pdist(df, 'euclidean')

with open ('dis_table_condensed.txt', 'w')  as f:
    l = len(dis_condensed)
    for k, el in enumerate(dis_condensed):
        f.write('{}\n'.format(el))
        print('Progress #1: {}/{}\n'.format(k, l))
        
# Write to file with format
N = df.loc[:, '0'].count()
i = 0
j = 0
with open ('dis_table_sparse.txt', 'w')  as f:
    l = len(dis_condensed)
    for k, el in enumerate(dis_condensed):
        if (i == j):
            f.write('{} {} {}\n'.format(i, j, str(0.0)))
            j += 1
        f.write('{} {} {}\n'.format(i, j, el))
        j += 1
        if j == N:
            i += 1
            j = i
        print('Progress #2: {}/{}\n'.format(k, l))
    f.write('{} {} {}\n'.format(i, j, str(0.0)))