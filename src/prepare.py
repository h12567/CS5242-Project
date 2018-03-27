import csv

MAX_LEN = 4096

with open('../data/actual_test.csv', 'w', newline='') as actualcsvfile:
    actualwriter = csv.writer(actualcsvfile)
    with open('../data/malware_indices.csv', 'w', newline='') as malwarecsvfile:
        malwarewriter = csv.writer(malwarecsvfile)
        with open('../data/test.csv') as csvfile:
            rowreader = csv.reader(csvfile)
            for index, row in enumerate(rowreader):
                if len(row) > MAX_LEN:
                    malwarewriter.writerow([1])
                else:
                    actualwriter.writerow(row)
                    malwarewriter.writerow([0])
