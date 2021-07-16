import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Conv1D, Reshape, MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
import _pickle as pickle

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

    model.load_weights("/home/xvpher/Intern_Project/Convolutional Neural Network/meta/model_weights.h5", by_name=True)

    pred = model.predict(data['X_train'])
    with open("/home/xvpher/Intern_Project/Convolutional Neural Network/svm_inputs/svm_train_input", 'wb') as outfile:
        pickle.dump(pred, outfile)

    pred = model.predict(data['X_test'])
    with open("/home/xvpher/Intern_Project/Convolutional Neural Network/svm_inputs/svm_test_input", 'wb') as outfile:
        pickle.dump(pred, outfile)


if __name__ == '__main__':
    main()
