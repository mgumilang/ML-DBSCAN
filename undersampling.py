with open ('CencusIncome.csv', 'r') as f:
    lines = f.readlines()

data_per_class = 5
with open ('CencusIncomeUndersampled.csv', 'w') as f:
    i = 0
    j = 0
    for line in lines:
        x = line.split(',')
        if (x[-1] == '<=50K\n') and (i < data_per_class):
            i += 1
            f.write(line)
        elif (x[-1] == '>50K\n') and (j < data_per_class):
            j += 1
            f.write(line)