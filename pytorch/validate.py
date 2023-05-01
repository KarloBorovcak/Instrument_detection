# import tensorflow as tf
# from tensorflow import keras
from torch import from_numpy
import onnxruntime as onnxrt
import numpy as np
from preprocessingivara import create_spectogram, split_file
import os

score = 0
total = 0

def load_models():
    
    loaded_model = onnxrt.InferenceSession("./models/model.onnx")
    loaded_model.set_providers(['CPUExecutionProvider'], [{'precision': 'float32'}])
    return loaded_model

def predict(model, data):
    global instruments, instrument_list, tresholds
    global precission, recall, accuracy, total, exacts, TP, FP, TN, FN
    # Predict
    total += 1
    pred_max = [0,0,0,0,0,0,0,0,0,0,0]    
    for signal in data:
        signal = np.resize(signal, (1, 128, 44))
        onnx_inputs= {model.get_inputs()[0].name: np.expand_dims(signal, 0)}
        prediction = model.run(None, onnx_inputs)
        prediction = prediction[0]
        for i in range(len(prediction[0])):
            if prediction[0][i] > pred_max[i]:
                pred_max[i] = prediction[0][i]
                if prediction[0][i] > tresholds[i]:
                    pred_max[i] = 1
                    continue


     
    print("pred_max: ", pred_max)    

    for i in range(len(pred_max)):
        if pred_max[i] > tresholds[i]:
            pred_max[i] = 1
            if instruments[i] == 1:
                TP += 1
            elif instruments[i] == 0:
                FP += 1
        else:
            pred_max[i] = 0
            if instruments[i] == 1:
                FN += 1
            elif instruments[i] == 0:
                TN += 1
            
    if pred_max == instruments:
        exacts += 1
    
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if TP + FP == 0:
        precission = 0
    else:
        precission = TP / (TP + FP)
    if precission + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precission * recall) / (precission + recall)

    print("Predciton: ", pred_max)
    print("Instruments: ", instruments)
    print("RECALL: ", recall) 
    print("ACCURACY: ", accuracy)
    print("PRECISION: ", precission) 
    print("F1: ", f1)         
    print("EXACT MATCHES: ", exacts/total)
    
            
    


if __name__ == "__main__":
    # Load model
    model = load_models()
    
    # Load data
    path = '../../DataLumenDS/Dataset/IRMAS_Validation_Data/'
    tresholds = [0.85, 0.85, 0.8, 0.5, 0.7, 0.7, 0.7, 0.85, 0.85, 0.8, 0.7]
    accuracy, recall, precission, total, exacts, TP, FP, TN, FN = 0, 0, 0, 0, 0, 0, 0, 0, 0
    instrument_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
    
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path[-3:] == "txt":
                instruments = [0,0,0,0,0,0,0,0,0,0,0]
                with open(file_path, "r") as f:
                    data = f.readlines()
                    for instrument in data:
                        if instrument.strip() in instrument_list:
                            instruments[instrument_list.index(instrument.strip())] = 1
                
              
                file_path = file_path[:-4] + ".wav"                
                # Split data
                split_signals = split_file(file_path)
                
                # Create spectrograms
                spectrograms = []
                for signal in split_signals.T:
                    spectrogram = create_spectogram(signal)
                    spectrograms.append(np.expand_dims(spectrogram, 2))
                
                
                # Predict
                predict(model, np.array(spectrograms))
                
            
 


    
    