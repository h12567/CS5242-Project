import csv

MAX_LEN = 4096

with open('../data/submit.csv', 'w', newline='') as submitcsvfile:
    submitwriter = csv.writer(submitcsvfile)
    submitwriter.writerow(["sample_id", "malware"])
    with open('../data/predict.csv') as predictcsvfile:
        predictreader = csv.reader(predictcsvfile)
        with open('../data/malware_indices.csv') as csvfile:
            rowreader = csv.reader(csvfile)
            for index, row in enumerate(rowreader):
                if row[0] == "1":
                    submitwriter.writerow([index, 1])
                else:
                    submitwriter.writerow([index] + predictreader.__next__())
