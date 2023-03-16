import librosa
import numpy as np

# Parameters

K = 1.5 # Duration of each sample in seconds
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

    # Calculate Mel Spectrogram
    n_fft = 1024
    n_mels = 128
    hop_size = 512
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=SR, n_fft=n_fft, hop_length=hop_size, n_mels=n_mels)

    # Convert Mel Spectrogram to MFCCs
    n_mfcc = 13
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)

    # Normalize MFCCs
    mfccs_normalized = (mfccs - np.mean(mfccs, axis=0)) / np.std(mfccs, axis=0)

    return mfccs_normalized



