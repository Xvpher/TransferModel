import os
import pandas as pd
import numpy as np
import _pickle as pickle

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Reshape, MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler

def main():
    os.chdir('../Dataset')
    path = os.getcwd()
    data = {}
    filenames = ["X_train", "X_test", "y_train", "y_test"]
    for i in range(len(filenames)):
        file = path+"/"+filenames[i]
        with open(file, 'rb') as infile:
            data[filenames[i]] = pickle.load(infile)
    model = Sequential()

    model.add(Conv1D(100, 10, activation='relu', input_shape=(57,1), name='conv1'))
    model.add(Conv1D(100, 10, activation='relu', name='conv2'))
    model.add(MaxPooling1D(3))

    model.add(Conv1D(160, 10, activation='relu', name='conv3'))
    model.add(GlobalAveragePooling1D())

    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name="Dense"))

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(data['X_train'],data['y_train'],batch_size=10,epochs=20,validation_split=0.2)

    os.chdir('../Convolutional Neural Network')
    path = os.getcwd()+"/meta"
    model.save_weights(path+"/model_weights.h5")

if __name__ == '__main__':
    main()
