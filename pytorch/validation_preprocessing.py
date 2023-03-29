import librosa
import numpy as np
import os

# Parameters

K = 1 # Duration of each sample in seconds
SR = 22050 # Sample rate

X, y = [], []
X_train, y_train = [], []
X_test, y_test = [], []

instruments = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

def split_file(data):
    """ Split audio file into samples of length K seconds """
    # Load data
    signal = librosa.load(data,
                          sr=SR,
                          mono=True)[0]
    
    # signal = signal / np.max(np.abs(signal))

    # Pad data
    # num_missing_items is the number of missing items to be added to the left of the signal
    num_missing_items = (SR * K) - len(signal) % (SR * K) 

    if num_missing_items == SR * K:
        num_missing_items = 0
    
    # Pad signal
    padded_signal = np.pad(signal,
                           (num_missing_items, 0),
                           mode="constant")
                           
    # Split data
    # frame_length is the number of samples in each frame
    # hop_length is the number of samples between the starts of consecutive frames
    frame_length = int(SR * K)
    split_signals = librosa.util.frame(padded_signal, frame_length=frame_length, hop_length=frame_length) 
                 
    return split_signals

def create_spectogram(signal):

    # Calculate Mel Spectrogram
    n_fft = 1024
    n_mels = 128
    hop_length = 512
    stft = librosa.feature.melspectrogram(y=signal, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    #normalized_spectrogram = (log_spectrogram - np.mean(log_spectrogram, axis=0)) / np.std(log_spectrogram, axis=0)
    
    # # Convert Mel Spectrogram to MFCCs
    # n_mfcc = 30
    # mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)

    # Normalize MFCCs
    #melspec_normalized = (log_mel_spec - np.mean(log_mel_spec, axis=0)) / np.std(log_mel_spec, axis=0)
    # print(mel_spec.shape)
    return log_spectrogram



