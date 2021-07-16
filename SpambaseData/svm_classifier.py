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
    filenames = ["X_train", "X_test", "y_train", "y_test"]
    for i in range(len(filenames)):
        file = path+"/"+filenames[i]
        with open(file, 'rb') as infile:
            data[filenames[i]] = pickle.load(infile)
    data['X_train'] = np.reshape(data['X_train'], (-1,57))
    data['X_test'] = np.reshape(data['X_test'], (-1,57))
    model = SVC(gamma='scale', kernel='rbf')
    model.fit(data['X_train'],data['y_train'])
    print (model.score(data['X_test'],data['y_test']))

if __name__ == '__main__':
    main()
