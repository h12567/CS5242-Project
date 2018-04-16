import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from operator import itemgetter
import keras
from keras.layers import *
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.regularizers import *
from keras.optimizers import *
from keras.callbacks import Callback, ModelCheckpoint


# Only read up to 4096 bytes, > 4096 has 100% malware rate
MAX_SIZE = 4096
TOTAL_ROWS =  113636
TOTAL_ROWS = 100000
USE_COLS = list(range(2, MAX_SIZE))
ROWS = TOTAL_ROWS

train = pd.read_csv("./data/train/train.zip", nrows=ROWS, usecols=USE_COLS, header=None, names = list(range(0, MAX_SIZE)), error_bad_lines=False)
train_label = pd.read_csv("./data/train_label.csv", usecols=[1], nrows=ROWS)

train = train.fillna(0, downcast='infer')
assert train.shape[0] == train_label.shape[0], "Train and label shapes are different"

# Save model
bgt = lgb.Booster(model_file="./model-99034.txt")

feature_indices = bgt.feature_name()
feature_importance = bgt.feature_importance()

feature_rank = []

for i, index in enumerate(feature_indices):
    importance = feature_importance[i]
    if importance == 0:
        continue
    
    feature_rank.append((index, importance))

feature_rank.sort(key=itemgetter(1), reverse=True)
    
mask_feature=np.fromiter(map(lambda x: x[0], feature_rank[0: 900]), dtype=np.int)

mask = np.random.rand(len(train)) < 0.9

train_data = train.values
train_labels = train_label.values
maxlen = 328

x_train = train_data[mask]
y_train = train_labels[mask]
x_test = train_data[~mask]
y_test = train_labels[~mask]

x_train = x_train[:, 0: mask_feature]
x_test = x_test[:, 0: mask_feature]


embed_size = 16
num_layers = 3
S = 256
lstm_layer_size = S
state_size = lstm_layer_size*num_layers

main_input = Input(shape=(maxlen,), dtype='int32',
                   name='main_input')
emb = Embedding(256, embed_size, input_length=maxlen,
                dropout=0.2, W_regularizer=l2(1e-4))(main_input)
hs = [] #hidden states from each LSTM layer stored here
hs.append(LSTM(S, dropout_W=0.5, dropout_U=0.5,
         W_regularizer=l2(1e-5), U_regularizer=l2(1e-5),
        return_sequences=True)(emb))

for l in range(1, num_layers):
    hs.append(LSTM(S, dropout_W=0.5, dropout_U=0.5,
              W_regularizer=l2(1e-5), U_regularizer=l2(1e-5),
              return_sequences=True)(hs[-1]))

print(len(hs))
print(hs[0]._keras_shape)
print(hs[1]._keras_shape)
print(hs[2]._keras_shape)

local_states = merge(hs, mode='concat')

print(local_states._keras_shape)

sum_dim1 = Lambda(lambda xin: K.mean(xin, axis=1))
average_active = sum_dim1(local_states)

print(average_active._keras_shape)

#Attention mechanism starts here
attn_cntx = merge([local_states,
                   RepeatVector(maxlen)(average_active)],
                   mode='concat')

attn_cntx = TimeDistributed(Dense(lstm_layer_size,
                            activation='linear',
                            W_regularizer=l2(1e-4)))(attn_cntx)

attn_cntx = TimeDistributed(BatchNormalization())(attn_cntx)

attn_cntx = TimeDistributed(Activation('tanh'))(attn_cntx)
attn_cntx = TimeDistributed(Dropout(0.5))(attn_cntx)

attn = TimeDistributed(Dense(1, activation='linear',
                             W_regularizer=l2(1e-4)))(attn_cntx) # αei

attn = Flatten()(attn)

attn = Activation('softmax')(attn) # αi
print(attn._keras_shape)

attn = Reshape((maxlen, 1))(attn)
print(attn._keras_shape)

sum_dim2 = Lambda(lambda x: K.repeat_elements(x, state_size, 2))
attn = sum_dim2(attn)
print(attn._keras_shape)

final_context = merge([attn, local_states], mode='mul')
print(final_context._keras_shape)

sum_dim3 = Lambda(lambda x: K.sum(x, axis=1))

final_context = sum_dim3(final_context)

final_context = Dense(state_size, activation='linear',
                      W_regularizer=l2(1e-4))(final_context)

final_context = BatchNormalization()(final_context)

final_context = Activation('tanh')(final_context)

final_context = Dropout(0.5)(final_context)

loss_out = Dense(1, activation='sigmoid',
                 name='loss_out')(final_context)

model = Model(input=[main_input], output=[loss_out])
optimizer = Adam(lr=0.001, clipnorm=1.0)
model.compile(optimizer, loss='binary_crossentropy', metrics=['acc'])

cb = ModelCheckpoint(filepath='lstm_model_sub.h5', monitor='acc', save_best_only=True)

print(model.summary())  # Summarize the model
model.fit(x_train, y_train, epochs=15, verbose=1, batch_size=64, callbacks=[cb])  # Fit the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)  # Evaluate the model
model.save('lstm_model.h5')
print('Accuracy: %0.3f' % accuracy)