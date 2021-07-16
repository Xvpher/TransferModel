import os
import time

import numpy as np
import pandas as pd
import _pickle as pickle

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split as tts

def create_data():
    path = os.getcwd()
    filenames = ["x", "y", "z"]

    # X Train values

    arr = []
    for file in filenames:
        inpath = path + "/Dataset/train/Inertial Signals/body_acc_"+file+"_train.txt"
        if not(os.path.exists(inpath)):
            print("Could not find the data file at - ", inpath)
            return
        with open(inpath, 'r') as infile:
            lines = infile.readlines()
            for line in lines:
                row = [float(x.strip()) for x in line.rstrip('\n').split()]
                arr.extend(row)
    arr = np.swapaxes(np.swapaxes(np.reshape(np.array(arr), (3,-1,128)),0,1),1,2)
    outpath = path + "/clean_data/X_train"
    with open(outpath, 'wb') as outfile:
        pickle.dump(arr, outfile)

    # Y Train values

    labels = []
    inpath = path + "/Dataset/train/y_train.txt"
    if not(os.path.exists(inpath)):
        print("Could not find the data file at - ", inpath)
        return
    with open(inpath, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            row = [int(x.strip()) for x in line.rstrip('\n').split()]
            labels.extend(row)
    labels = np.reshape(np.array(labels), (-1,1))
    onehot = OneHotEncoder(sparse=False, categories='auto')
    labels = onehot.fit_transform(labels)
    outpath = path + "/clean_data/y_train"
    with open(outpath, 'wb') as outfile:
        pickle.dump(labels, outfile)

    # X Test values

    arr = []
    for file in filenames:
        inpath = path + "/Dataset/test/Inertial Signals/body_acc_"+file+"_test.txt"
        if not(os.path.exists(inpath)):
            print("Could not find the data file at - ", inpath)
            return
        with open(inpath, 'r') as infile:
            lines = infile.readlines()
            for line in lines:
                row = [float(x.strip()) for x in line.rstrip('\n').split()]
                arr.extend(row)
    arr = np.swapaxes(np.swapaxes(np.reshape(np.array(arr), (3,-1,128)),0,1),1,2)
    outpath = path + "/clean_data/X_test"
    with open(outpath, 'wb') as outfile:
        pickle.dump(arr, outfile)

    # Y Test values

    labels = []
    inpath = path + "/Dataset/test/y_test.txt"
    if not(os.path.exists(inpath)):
        print("Could not find the data file at - ", inpath)
        return
    with open(inpath, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            row = [int(x.strip()) for x in line.rstrip('\n').split()]
            labels.extend(row)
    labels = np.reshape(np.array(labels), (-1,1))
    onehot = OneHotEncoder(sparse=False, categories='auto')
    labels = onehot.fit_transform(labels)
    outpath = path + "/clean_data/y_test"
    with open(outpath, 'wb') as outfile:
        pickle.dump(labels, outfile)

    print("------------------------------------Data extracted and saved in clean_data folder------------------------")
    return


def view_data():
    path = os.getcwd()
    filenames = ["X_train", "X_test", "y_train", "y_test"]
    for files in filenames:
        inpath = path + "/clean_data/"+files
        with open(inpath, 'rb') as infile:
            X_train = pickle.load(infile)
        print(X_train.shape)

def create_model():
    model = Sequential()
    model.add(Conv1D(100, 10, activation='relu', input_shape=(128,3), name='conv1'))
    model.add(Conv1D(100, 10, activation='relu', name='conv2'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(160, 10, activation='relu', name='conv3'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax', name="Dense"))
    print("------------------------------------Sequential model created---------------------------------------------")
    return(model)

def compile_model(model):
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    print("------------------------------------Model compiled-------------------------------------------------------")
    return(model)

def fit_model(model, X_train, y_train):
    model.fit(X_train,y_train,batch_size=10,epochs=20,validation_split=0.2)
    print("------------------------------------Model fitted with train values---------------------------------------")
    return(model)

def evaluate_model(model, X_test, y_test):
    print("------------------------------------accuracy of CNN------------------------------------------------------")
    print(model.evaluate(X_test,y_test))
    return

def save_model(model):
    path = "/home/xvpher/Intern_Project/Convolutional Neural Network/meta/model_weights.h5"
    model.save_weights(path)
    print("------------------------------------model saved----------------------------------------------------------")
    return

def create_transfer_model():
    model = Sequential()
    model.add(Conv1D(100, 10, activation='relu', input_shape=(128,3), name='conv1'))
    model.add(Conv1D(100, 10, activation='relu', name='conv2'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(160, 10, activation='relu', name='conv3'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.load_weights("/home/xvpher/Intern_Project/Convolutional Neural Network/meta/model_weights.h5", by_name=True)
    print("------------------------------------transfer model created with trained model weights--------------------")
    return(model)

def create_svm_inputs(model, X_train, X_test):
    # Output of the CNN which acts as the input to the SVM Classifiers
    svm_X_train = model.predict(X_train)
    with open("/home/xvpher/Intern_Project/Convolutional Neural Network/svm_inputs/svm_X_train", 'wb') as outfile:
        pickle.dump(svm_X_train, outfile)
    svm_X_test = model.predict(X_test)
    with open("/home/xvpher/Intern_Project/Convolutional Neural Network/svm_inputs/svm_X_test", 'wb') as outfile:
        pickle.dump(svm_X_test, outfile)
    print("------------------------------------Inputs for the SVM Classifier created using transfer model-----------")
    return

def train_SVM(svm_X_train, y_train):
    model = SVC(gamma='scale', kernel='rbf')
    model.fit(svm_X_train,y_train)
    print("------------------------------------SVM trained----------------------------------------------------------")
    return(model)

def predict_SVM(model, svm_X_test, y_test):
    print("------------------------------------accuracy of the model------------------------------------------------")
    print (model.score(svm_X_test,y_test))
    return

def SVM_classifier(X_train, y_train, X_test, y_test):
    model = SVC(gamma='scale', kernel='rbf')
    model.fit(X_train,y_train)
    print("------------------------------------accuracy of simple SVM classifier------------------------------------")
    print (model.score(X_test,y_test))
    return

def driver():
    create_data()
    # view_data()

    data = {}
    path = os.getcwd()
    filenames = ["X_train", "X_test", "y_train", "y_test"]
    for i in range(len(filenames)):
        inpath = path+"/clean_data/"+filenames[i]
        with open(inpath, 'rb') as infile:
            data[filenames[i]] = pickle.load(infile)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    train_model = create_model()
    train_model = compile_model(train_model)
    train_model = fit_model(train_model, X_train, y_train)
    evaluate_model(train_model, X_test, y_test)
    # save_model(train_model)
    # transfer_model = create_transfer_model()
    # create_svm_inputs(transfer_model, X_train, X_test)
    #
    # path = "/home/xvpher/Intern_Project/Convolutional Neural Network/svm_inputs/"
    # filenames = ["svm_X_train", "svm_X_test"]
    # for i in range(len(filenames)):
    #     file = path+filenames[i]
    #     with open(file, 'rb') as infile:
    #         data[filenames[i]] = pickle.load(infile)
    #
    # svm_X_train = data['svm_X_train']
    # svm_X_test = data['svm_X_test']
    #
    # svm_model = train_SVM(svm_X_train, y_train)
    # predict_SVM(svm_model, svm_X_test, y_test)
    #
    # X_train = np.reshape(X_train, (-1,57))
    # X_test = np.reshape(X_test, (-1,57))
    #
    # SVM_classifier(X_train, y_train, X_test, y_test)

    print("Driver exhauted")
    return

if __name__ == '__main__':
    driver()
