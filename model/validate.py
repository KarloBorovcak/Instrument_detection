import tensorflow as tf
from tensorflow import keras
import numpy as np
from preprocessing import create_spectogram, split_file
import os

score = 0
total = 0

def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    
    return loaded_model

def predict(model, data):
    global instruments
    global total
    global score
    total += 1
    # Predict
    
    pred_max = 0    
    for signal in data:
        prediction = model.predict(np.expand_dims(signal, 0), verbose=0)
        if prediction[0][0] > pred_max:
            pred_max = prediction[0][0]
        if prediction[0][0] > 0.9:
            if instruments["cel"] == 1:
                score += 1
                return
            
    if instruments["cel"] == 0:
        score += 1
        return
    
    if instruments["cel"] == 1:
        print("Sigurnost: " + str(pred_max))
        return
    
 
            
    


if __name__ == "__main__":
    # Load model
    model = load_model()
    
    # Load data
    path = '../../DataLumenDS/Dataset/IRMAS_Validation_Data/'
    instruments = {"cel": 0, "cla": 0, "flu": 0, "gac": 0, "gel": 0, "org": 0, "pia": 0, "sax": 0, "tru": 0, "vio": 0, "voi": 0}
    
    
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path[-3:] == "txt":
                instruments = {"cel": 0, "cla": 0, "flu": 0, "gac": 0, "gel": 0, "org": 0, "pia": 0, "sax": 0, "tru": 0, "vio": 0, "voi": 0}
                with open(file_path, "r") as f:
                    data = f.readlines()
                    for instrument in data:
                        if instrument.strip() in instruments:
                            instruments[instrument.strip()] = 1
              
            if file_path[-3:] == "wav":
                if instruments["cel"] == 0:
                    continue
                # Split data
                split_signals = split_file(file_path)
                
                # Create spectrograms
                spectrograms = []
                for signal in split_signals.T:
                    spectrogram = create_spectogram(signal)
                    spectrograms.append(np.expand_dims(spectrogram, 2))
                
                
                # Predict
                prediction = predict(model, np.array(spectrograms))
                
                print(score/total)
 


    
    