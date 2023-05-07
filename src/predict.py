import librosa
import numpy as np
import json
import onnxruntime as onnxrt
import config
import os


def split_file(data):
    # Load data
    y = librosa.load(data, sr=config.SAMPLE_RATE, mono=True)[0]

    # Pad data if necessary
    num_missing_samples = (config.SAMPLE_RATE * config.DURATION) - len(y) % (config.SAMPLE_RATE * config.DURATION)

    if num_missing_samples != (config.SAMPLE_RATE * config.DURATION):
            y = np.pad(y, (num_missing_samples, 0), 'constant')
   
    # Split data
    frame_length = int(config.SAMPLE_RATE * config.DURATION)
    split_signals = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length) 
                 
    return split_signals

def create_spectogram(signal):
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, n_mels=config.N_MELS, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)

    # Compress with natural logarithm
    log_mel_spec = librosa.amplitude_to_db(np.abs(mel_spec))

    # Normalize
    normalized = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())

    return normalized


def predict(data):
    signals = split_file(data)
    loaded_model = onnxrt.InferenceSession("./model/models/model_densenet_121.onnx")
    loaded_model.set_providers(['CPUExecutionProvider'], [{'precision': 'float32'}])

    pred_max = [0] * len(config.INSTRUMENTS)
    instrument_dict = dict(zip(config.INSTRUMENTS, [0] * len(config.INSTRUMENTS)))

    for signal in signals.T:
        signal = create_spectogram(signal)
        signal = np.resize(signal, (1, 128, 44))
        onnx_inputs = {loaded_model.get_inputs()[0].name: np.expand_dims(signal, 0)}
        prediction = loaded_model.run(None, onnx_inputs)
        prediction = prediction[0]

        for i in range(len(prediction[0])):
            if prediction[0][i] > pred_max[i]:
                pred_max[i] = prediction[0][i]
                if prediction[0][i] > config.THRESHOLD:
                    pred_max[i] = 1
                    continue

        for i in range(len(pred_max)):
            if pred_max[i] > config.THRESHOLD:
                instrument_dict[config.INSTRUMENTS[i]] = 1
    
    return instrument_dict


if __name__ == "__main__":

    with open(config.TEST_DATA_PATH + "example_labels.json") as json_file:
        labels = json.load(json_file)
    files = os.listdir(config.TEST_DATA_PATH) 
    total = len(files)-1
    zeros = 0
    two = 0
    more = 0
    predictions = {}
    for file in files:
        if file.endswith(".wav"):
            print(file[:-4])
            prediction = predict(config.TEST_DATA_PATH + file)
            print(prediction)
            predictions[file[:-4]] = prediction
            if sum(prediction.values()) == 0:
                zeros += 1
            if sum(prediction.values()) == 2:
                two += 1
            if sum(prediction.values()) > 2:
                more += 1
            print()
        
    print("Total: ", total)
    print("Zeros: ", zeros)
    print("Two: ", two)
    print("More: ", more) 
    with open("predictions.json", 'w') as outfile:
        json.dump(predictions, outfile)
