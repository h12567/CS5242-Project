# Import all data
import lightgbm as lgb
from operator import itemgetter

# Save model
bgt = lgb.Booster(model_file="../model - 99034.txt")

feature_indices = bgt.feature_name()
feature_importance = bgt.feature_importance()

feature_rank = []

for i, index in enumerate(feature_indices):
    importance = feature_importance[i]
    if importance == 0:
        continue
    
    feature_rank.append((index, importance))

feature_rank.sort(key=itemgetter(1), reverse=True)
print("Number of useful features: ", len(feature_rank))
print("Index | Times used")
for index, feature in feature_rank:
    print('{:4} {:d}'.format(index, feature))
