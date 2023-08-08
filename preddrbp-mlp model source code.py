import pandas as pd

df = pd.read_csv(r"benchmark_dataset.csv", index_col = [0])
print(df.shape)

import numpy as np

X_train = df.drop(["Entry", "Type", "Sequence", "Length"], axis = 1).to_numpy()
y_train = df["Type"]

from sklearn.preprocessing import OneHotEncoder

categorical_encoding = {"nnabp": 0, "dbp": 1, "rbp": 2}
integer_encoded = np.array(y_train.replace(categorical_encoding))

one_hot_encoder = OneHotEncoder(sparse = False)
y_train = one_hot_encoder.fit_transform(integer_encoded.reshape(len(integer_encoded), 1))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, GaussianNoise
from keras.optimizers import Adam

model = Sequential()
model.add(Flatten(input_shape = (100, 1)))
model.add(Dense(200, activation = 'selu'))
model.add(GaussianNoise(0.1))
model.add(Dropout(0.5))
model.add(Dense(200, activation = 'selu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation = 'softmax'))
model.summary()

optimizer = Adam(learning_rate = 0.0001)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 128, epochs = 40, verbose = 0)
