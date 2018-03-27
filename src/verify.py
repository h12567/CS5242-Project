import csv

cols = []
length = []

with open('../data/train.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    max_len = 0
    for index, row in enumerate(spamreader):
        if len(row) >= max_len:
            cols.append(index)
            length.append(len(row))
            max_len = len(row)

with open('../data/train_label.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    count = 0
    for index, row in enumerate(spamreader):
        if (index - 1) in cols:
            print(index, length[count], row)
            count += 1