import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.layers import Conv1D, Reshape, MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler

def main():
    if not(os.path.exists("/home/xvpher/Intern_Project/Dataset/spamdata.csv")):
        print("Could not find the data file")
        return
    df = pd.read_csv("/home/xvpher/Intern_Project/Dataset/spamdata.csv")
    features = df.iloc[:,0:57].values
    labels = df.iloc[:,57].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = tts(features, labels, test_size=0.2, shuffle=True, random_state=8)
    X_train = np.reshape(X_train, (-1,57,1))
    X_test = np.reshape(X_test, (-1,57,1))
    count=0
    model = tf.keras.models.load_model("/home/xvpher/Intern_Project/Convolutional Neural Network/model_1.model")
    for test in X_test:
        test = np.reshape(test, (-1,57,1))
        prediction = model.predict(test)
        print (prediction)
        count=count+1
        if (count==10):
            break
    layer = model.get_layer("Dense")
    # print (model.evaluate(X_test, y_test))
    # print (model.get_weights())
    # print(layer.get_weights())

if __name__ == '__main__':
    main()
