import pandas as pd
import numpy as np

# Only read up to 4096 bytes, > 4096 has 100% malware rate
MAX_SIZE = 4096
TOTAL_ROWS = 113636
USE_COLS = list(range(2, MAX_SIZE))
ROWS = TOTAL_ROWS

train = pd.read_csv("./train.csv", nrows=ROWS, usecols=USE_COLS, header=None, names = list(range(0, MAX_SIZE)), error_bad_lines=False)
train_label = pd.read_csv("./train_label.csv", usecols=[1], nrows=ROWS).values

train = np.nan_to_num(train, copy=False)
assert train.shape[0] == train_label.shape[0], "Train and label shapes are different"

import keras
from keras.layers import *
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer

mask = np.random.rand(len(train)) < 0.8

x_train = train[mask]
y_train = train_label[mask]
x_test = train[~mask]
y_test = train_label[~mask]

del train
del train_label

class Attention(Layer):

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        multData =  K.dot(x, self.kernel) 
        multData = K.squeeze(multData, -1)
        multData = multData + self.b 

        multData = K.tanh(multData) 

        multData = multData * self.u 
        multData = K.exp(multData) 

        if mask is not None:
            mask = K.cast(mask, K.floatx()) 
            multData = mask*multData 

        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)

own_embedding_vocab_size = 256

maxlen = 4094

model = Sequential()
model.add(Embedding(input_dim=own_embedding_vocab_size, # 10
                    output_dim=32, 
                    input_length=maxlen))
model.add(Dropout(rate=0.25))

model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32, return_sequences=True))

model.add(Attention())
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))

model.add(Dense(1, activation='sigmoid'))

adam=optimizers.Adam(lr=0.01)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])  # Compile the model
print(model.summary())  # Summarize the model
model.fit(x_train, y_train, epochs=15, verbose=1, batch_size=64)  # Fit the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)  # Evaluate the model
model.save('lstm_model.h5')
print('Accuracy: %0.3f' % accuracy)
