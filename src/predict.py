import pandas as pd
import numpy as np
from keras.models import load_model

# Only read up to 4096 bytes, > 4096 has 100% malware rate
MAX_SIZE = 4096
USE_COLS = list(range(2, MAX_SIZE))
TOTAL_ROWS = 133223
ROWS = TOTAL_ROWS

model = load_model('../mlp_model.h5')

test = pd.read_csv("../data/test.csv", nrows=ROWS, usecols=USE_COLS, header=None, names = list(range(0, MAX_SIZE)))
test = np.nan_to_num(test, copy=False)
ypred = model.predict(test)

df = pd.DataFrame({'sample_id': range(len(test)), 'malware': ypred.flatten()}, columns=['sample_id', 'malware'])
df.to_csv('../data/predict.csv', index=False)
