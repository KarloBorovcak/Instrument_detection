import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

instruments = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

path = '../DataLumenDS/Processed/'

df = pd.DataFrame(columns=['data', 'label'])

for instrument in instruments:
    if instrument == 'cel' or instrument == 'cla':
        continue
    df = pd.DataFrame(columns=['data', 'label'])
    for instrumentdir in instruments:
        for root, _, files in os.walk(path + instrumentdir):
            for file in files:
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                data = np.array(data)
                data = data.reshape((data.shape[0], data.shape[1], 1))
                temp = pd.DataFrame({'data': [data], 'label': 1 if instrumentdir == instrument  else 0})
                df = pd.concat([df, temp])


    temp_cellist = df[df['label'] == 1]
    temp_non_cellist = df[df['label'] == 0]

    X_train, X_test, y_train, y_test = train_test_split(temp_cellist['data'], temp_cellist['label'], test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(temp_non_cellist['data'], temp_non_cellist['label'], test_size=0.2, random_state=42)


    x = len(X_train)
    y = len(y_train)
    print(x, y)

    X_train = pd.concat([X_train, X_train2[:x]])
    print(len(X_train))
    X_test = pd.concat([X_test, X_test2])

    y_train = pd.concat([y_train, y_train2[:y]])
    print(len(y_train))
    y_test = pd.concat([y_test, y_test2])
    
    X_train = np.stack(X_train)
    X_test = np.stack(X_test)
    y_train = np.stack(y_train)
    y_test = np.stack(y_test)

    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(513, 33, 1)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile('Adam', loss='BinaryCrossentropy')
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    # serialize model to JSON
    model_json = model.to_json()
    with open("./model/models/model_" + instrument + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./model/models/model_" + instrument + ".h5")
    print("Saved model to disk")