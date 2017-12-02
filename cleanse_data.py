#%%
with open ("CensusIncome/CencusIncome.data.txt", 'r') as f:
    lines = f.readlines()
    
with open ("CencusIncome.csv", 'w') as f:
    for line in lines:
        f.write(line.replace(', ', ','))
        
#%%
with open ("CensusIncome/CencusIncome.test.txt", 'r') as f:
    lines = f.readlines()
    
with open ("CencusIncome.test.csv", 'w') as f:
    for line in lines:
        f.write(line.replace(', ', ','))
