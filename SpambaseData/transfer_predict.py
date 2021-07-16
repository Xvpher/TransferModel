import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
import _pickle as pickle


def main():
    os.chdir('../Dataset')
    path = os.getcwd()
    data = {}
    filenames = ["y_train", "y_test"]
    for i in range(len(filenames)):
        file = path+"/"+filenames[i]
        with open(file, 'rb') as infile:
            data[filenames[i]] = pickle.load(infile)
    os.chdir('../Convolutional Neural Network/svm_inputs')
    path = os.getcwd()
    filenames = ["svm_train_input", "svm_test_input"]
    for i in range(len(filenames)):
        file = path+"/"+filenames[i]
        with open(file, 'rb') as infile:
            data[filenames[i]] = pickle.load(infile)

    model = SVC(gamma='scale', kernel='rbf')
    model.fit(data['svm_train_input'],data['y_train'])
    pred = model.predict(data['svm_test_input'])
    print (model.score(data['svm_test_input'],data['y_test']))

if __name__ == '__main__':
    main()
