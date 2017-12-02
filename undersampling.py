with open ('CencusIncome.csv', 'r') as f:
    lines = f.readlines()
    
with open ('CencusIncomeUndersampled.csv', 'w') as f:
    i = 0
    for line in lines:
        x = line.split(',')
        if (x[-1] == '<=50K\n') and (i < 7841):
            i += 1
            f.write(line)
        elif (x[-1] == '>50K\n'):
            f.write(line)