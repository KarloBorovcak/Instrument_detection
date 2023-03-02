import librosa
import numpy as np

# Parameters

K = 0.75 # Duration of each sample in seconds
SR = 22050 # Sample rate

def split_file(data):
    """ Split audio file into samples of length K seconds """
    # Load data
    signal = librosa.load(data,
                          sr=SR,
                          mono=True)[0]
    
    # Pad data
    # num_missing_items is the number of missing items to be added to the left of the signal
    num_missing_items = (SR * K) - len(signal) % (SR * K) 

    if num_missing_items == SR * K:
        num_missing_items = 0
    
    # Pad signal
    padded_signal = np.pad(signal,
                           (int(num_missing_items), 0),
                           mode="constant")
                           
    # Split data
    # frame_length is the number of samples in each frame
    # hop_length is the number of samples between the starts of consecutive frames
    frame_length = int(SR * K)
    split_signals = librosa.util.frame(padded_signal, frame_length=frame_length, hop_length=frame_length) 
                        
    return split_signals

def create_spectogram(signal):

    # Extract log spectrogram
    stft = librosa.stft(signal,
                        n_fft=1024,
                        hop_length=512)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    normalised_log_spectrogram = (log_spectrogram - log_spectrogram.min()) / (log_spectrogram.max() - log_spectrogram.min())

    return normalised_log_spectrogram



