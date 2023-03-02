import librosa
import numpy as np
import os

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
                           (num_missing_items, 0),
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
    save = "../DataLumenDS/Processed/"
    data = "../DataLumenDS/Dataset/IRMAS_Training_Data/"
    instruments = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

    for instrument in instruments:
        save_path = os.path.join(save, instrument)
        data_path = os.path.join(data, instrument)

        for root, _, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)

                # Save spectrograms
                save_spectogram(file_path, save_path)

        



