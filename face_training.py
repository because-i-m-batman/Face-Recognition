from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import Sequential
from keras.layers import Dense
from keras import Input
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint




data = load('Path to face embeddings')
train_x,train_y = data['arr_0'],data['arr_1']

#create one hot encodings for labels
out_encoder = LabelEncoder()
out_encoder.fit(train_y)
trainy = out_encoder.transform(train_y)
new_train_y = np.zeros((trainy.size, trainy.max()+1))
new_train_y[np.arange(trainy.size),trainy] = 1

print(new_train_y)
exit()
#model training for embeddings
model = Sequential()
model.add(Input(shape = (128,)))
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(3,activation = 'softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

checkpoint = ModelCheckpoint('model.hdf5', monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(train_x, new_train_y, epochs=15, batch_size=4,callbacks = callbacks_list)
