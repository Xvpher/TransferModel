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
from sklearn.model_selection import train_test_split as tts

def create_data():
    datafile_path = "/home/xvpher/Intern_Project/Dataset/spamdata.csv"
    if not(os.path.exists(datafile_path)):
        print("Could not find the data file")
        return
    df = pd.read_csv(datafile_path)
    features = df.iloc[:,0:57].values
    labels = df.iloc[:,57].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = tts(features, labels, test_size=0.2, shuffle=True, random_state=165)
    X_train = np.reshape(X_train, (-1,57,1))
    X_test = np.reshape(X_test, (-1,57,1))
    var_names = [X_train, X_test, y_train, y_test]
    filenames = ["X_train", "X_test", "y_train", "y_test"]
    for i in range(len(var_names)):
        file = "/home/xvpher/Intern_Project/Dataset/"+filenames[i]
        with open(file, 'wb') as outfile:
            pickle.dump(var_names[i], outfile)
    print("------------------------------------Data extracted and saved in Dataset folder---------------------------")
    return

def create_model():
    model = Sequential()
    model.add(Conv1D(100, 10, activation='relu', input_shape=(57,1), name='conv1'))
    model.add(Conv1D(100, 10, activation='relu', name='conv2'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(160, 10, activation='relu', name='conv3'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name="Dense"))
    print("------------------------------------Sequential model created---------------------------------------------")
    return(model)

def compile_model(model):
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
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
    model.add(Conv1D(100, 10, activation='relu', input_shape=(57,1), name='conv1'))
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

    data = {}
    path = "/home/xvpher/Intern_Project/Dataset/"
    filenames = ["X_train", "X_test", "y_train", "y_test"]
    for i in range(len(filenames)):
        file = path+filenames[i]
        with open(file, 'rb') as infile:
            data[filenames[i]] = pickle.load(infile)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    train_model = create_model()
    train_model = compile_model(train_model)
    train_model = fit_model(train_model, X_train, y_train)
    evaluate_model(train_model, X_test, y_test)
    save_model(train_model)
    transfer_model = create_transfer_model()
    create_svm_inputs(transfer_model, X_train, X_test)

    path = "/home/xvpher/Intern_Project/Convolutional Neural Network/svm_inputs/"
    filenames = ["svm_X_train", "svm_X_test"]
    for i in range(len(filenames)):
        file = path+filenames[i]
        with open(file, 'rb') as infile:
            data[filenames[i]] = pickle.load(infile)

    svm_X_train = data['svm_X_train']
    svm_X_test = data['svm_X_test']

    svm_model = train_SVM(svm_X_train, y_train)
    predict_SVM(svm_model, svm_X_test, y_test)

    X_train = np.reshape(X_train, (-1,57))
    X_test = np.reshape(X_test, (-1,57))

    SVM_classifier(X_train, y_train, X_test, y_test)

    print("Driver exhauted")
    return

if __name__ == '__main__':
    driver()
