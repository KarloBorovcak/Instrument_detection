import librosa
import numpy as np
import json
import onnxruntime as onnxrt


DURATION = 1 # Duration of each sample in seconds
SAMPLE_RATE = 22050 # Sample rate
N_MELS = 128 # Number of mel bands to generate
N_FFT = 1024 # Length of the FFT window
HOP_LENGTH = 512 # Number of samples between successive frames
INSTRUMENT_LIST = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"] # List of instruments
THRESHOLD = 0.99 # Threshold for prediction


def split_file(data):
    # Load data
    y = librosa.load(data, sr=SAMPLE_RATE, mono=True)[0]

    # Pad data if necessary
    num_missing_samples = (SAMPLE_RATE * DURATION) - len(y) % (SAMPLE_RATE * DURATION)

    if num_missing_samples != (SAMPLE_RATE * DURATION):
            y = np.pad(y, (num_missing_samples, 0), 'constant')
   
    # Split data
    frame_length = int(SAMPLE_RATE * DURATION)
    split_signals = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length) 
                 
    return split_signals

def create_spectogram(signal):
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Compress with natural logarithm
    log_mel_spec = librosa.amplitude_to_db(np.abs(mel_spec))

    # Normalize
    normalized = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())

    return normalized


def predict(data):
    signals = split_file(data)
    loaded_model = onnxrt.InferenceSession("./model/model.onnx")
    loaded_model.set_providers(['CPUExecutionProvider'], [{'precision': 'float32'}])

    pred_max = [0] * len(INSTRUMENT_LIST)
    instrument_dict = dict(zip(INSTRUMENT_LIST, [0] * len(INSTRUMENT_LIST)))

    for signal in signals.T:
        signal = create_spectogram(signal)
        signal = np.resize(signal, (1, 128, 44))
        onnx_inputs = {loaded_model.get_inputs()[0].name: np.expand_dims(signal, 0)}
        prediction = loaded_model.run(None, onnx_inputs)
        prediction = prediction[0]

        for i in range(len(prediction[0])):
            if prediction[0][i] > pred_max[i]:
                pred_max[i] = prediction[0][i]
                if prediction[0][i] > THRESHOLD:
                    pred_max[i] = 1
                    continue

        for i in range(len(pred_max)):
            if pred_max[i] > THRESHOLD:
                instrument_dict[INSTRUMENT_LIST[i]] = 1
    
    return json.dumps(instrument_dict)
        
