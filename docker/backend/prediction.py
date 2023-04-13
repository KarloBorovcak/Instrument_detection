import librosa
import numpy as np
import json
import onnxruntime as onnxrt

K = 1 # Duration of each sample in seconds
SR = 22050 # Sample rate

def split_file(data):
    """ Split audio file into samples of length K seconds """
    # Load data
    y = librosa.load(data, sr=SR, mono=True)[0]

    num_missing_samples = (SR * K) - len(y) % (SR * K)

    if num_missing_samples == (SR * K):
        num_missing_samples = 0

    padded_signal = np.pad(y, (num_missing_samples, 0), 'constant')
   
    # Split data
    # frame_length is the number of samples in each frame
    # hop_length is the number of samples between the starts of consecutive frames
    frame_length = int(SR * K)
    split_signals = librosa.util.frame(padded_signal, frame_length=frame_length, hop_length=frame_length) 
                 
    return split_signals

def create_spectogram(signal):

    # Compute STFT
    n_fft = 1024
    hop_length = 512
    # stft = librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length)

    # Convert to Mel scale
    n_mels = 128
    mel_spec = librosa.feature.melspectrogram(y=signal, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    # Compress with natural logarithm
    log_mel_spec = librosa.amplitude_to_db(np.abs(mel_spec))

    normalized = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())


    return normalized


def predict(data):
    signals = split_file(data)
    loaded_model = onnxrt.InferenceSession("model.onnx")

    pred_max = [0,0,0,0,0,0,0,0,0,0,0]
    instrument_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
    instrument_dict = {"cel": 0, "cla": 0, "flu": 0, "gac": 0, "gel": 0, "org": 0, "pia": 0, "sax": 0, "tru": 0, "vio": 0, "voi": 0}

    for signal in signals.T:
        signal = create_spectogram(signal)
        signal = np.resize(signal, (1, 128, 44))
        onnx_inputs= {loaded_model.get_inputs()[0].name: np.expand_dims(signal, 0)}
        prediction = loaded_model.run(None, onnx_inputs)
        prediction = prediction[0]
        for i in range(len(prediction[0])):
            if prediction[0][i] > pred_max[i]:
                pred_max[i] = prediction[0][i]
                if prediction[0][i] > 0.8:
                    pred_max[i] = 1
                    continue

    for i in range(len(pred_max)):
        if pred_max[i] > 0.8:
            instrument_dict[instrument_list[i]] = 1
    
    return json.dumps(instrument_dict)
        
