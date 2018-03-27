# Import all data
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

# Only read up to 4096 bytes, > 4096 has 100% malware rate
MAX_SIZE = 4096
TOTAL_ROWS = 113636
USE_COLS = list(range(2, MAX_SIZE))
ROWS = TOTAL_ROWS

train = pd.read_csv("./data/train.zip", nrows=ROWS, usecols=USE_COLS, header=None, names = list(range(0, MAX_SIZE)))
train_label = pd.read_csv("./data/train_label.csv", usecols=[1], nrows=ROWS)

# train = train.fillna(0, downcast='infer')
assert train.shape[0] == train_label.shape[0], "Train and label shapes are different"

mask = np.random.rand(len(train)) < 0.8

x_train = train[mask]
y_train = train_label[mask]
x_test = train[~mask]
y_test = train_label[~mask]
train_data = lgb.Dataset(x_train, label=y_train.values.ravel())

# Create validation data
test_data = train_data.create_valid(x_test, label=y_test.values.ravel())

train = None
x_train = None
y_train = None
x_test = None
y_test = None
gc.collect()

params = {
    'learning_rate': 0.03,
    'num_leaves': 51, 
    'lambda_l2': 0.01,
    'objective':'binary',
    'tree_learner': 'voting_parallel',
    'bagging_freq': 10,
    'early_stopping_rounds': 10,
    'top_k': 35,
}
num_round = 2000
bst = lgb.train(params, 
                train_data, 
                num_round, 
                valid_sets=[test_data], 
                init_model='model.txt',
               )
# Save model
bst.save_model('model.txt', num_iteration=bst.best_iteration)

TOTAL_ROWS = 133223
ROWS = TOTAL_ROWS
test = pd.read_csv("./data/actual_test.csv", nrows=ROWS, usecols=USE_COLS, header=None, names = list(range(0, MAX_SIZE)))
ypred = bst.predict(test, num_iteration=bst.best_iteration)

df = pd.DataFrame(ypred)
df.to_csv('./data/predict.csv', index=False, header=False)
