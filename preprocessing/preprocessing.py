import librosa
import numpy as np
import os

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
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert Mel Spectrogram to MFCCs
    n_mfcc = 13
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)

    # Normalize MFCCs
    mfccs_normalized = (mfccs - np.mean(mfccs, axis=0)) / np.std(mfccs, axis=0)

    return mfccs_normalized

def save_spectogram(data, path):
    """ Save spectrogram of each sample of the audio file """
    # Split data
    split_signals = split_file(data)

    # Save spectrograms
    for i, signal in enumerate(split_signals.T):
        log_spectrogram = create_spectogram(signal)

        file_name = os.path.split(data)[1]
        save_path = os.path.join(path, file_name + f"_{i}.npy")
        np.save(save_path, log_spectrogram)


if __name__ == "__main__":
    # Load data
    SAVE_DIR = "../DataLumenDS/Processed/"
    DATA_DIR = "../DataLumenDS/Dataset/IRMAS_Training_Data/"
    instruments = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

    for instrument in instruments:
        
        save_path = os.path.join(SAVE_DIR, instrument)
        data_path = os.path.join(DATA_DIR, instrument)

        for root, _, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)

                # Save spectrograms
                save_spectogram(file_path, save_path)

        



