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


noise_factor = 0.08
def augment_data(signal):
    noise = np.random.randn(len(signal))
    noised_signal = signal + noise_factor * noise

    pitch_shifted = librosa.effects.pitch_shift(signal, sr=SR, n_steps=2)

    return [noised_signal, pitch_shifted]

cnt = 0
SKIP = 6
def save_spectogram(data, path, instrument):
    global cnt
    """ Save spectrogram of each sample of the audio file """
    # Split data
    cnt += 1
    split_signals = split_file(data)
    tempy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tempy[instruments.index(instrument)] = 1
    
    # if cnt%SKIP==0:
    for i, signal in enumerate(split_signals.T):
        log_spectrogram = create_spectogram(signal)
        file_name = os.path.split(data)[1]
        save_path = os.path.join(path, file_name + f"_{i}.npy")

        np.save(save_path, log_spectrogram)
        
    # else:
    #     for i, signal in enumerate(split_signals.T):
    #         log_spectrogram = create_spectogram(signal)
    #         X_train.append(log_spectrogram)
    #         y_train.append(tempy.copy())
            # augmented_data = augment_data(signal)
            # for j in augmented_data: # adding augmented data to train set
            #     log_spectrogram = create_spectogram(j)
            #     X_train.append(log_spectrogram)
            #     y_train.append(tempy.copy())
    
            
        




if __name__ == "__main__":
    # Load data
    cnter=0
    SAVE_DIR = "../DataLumenDS/Processed/"
    DATA_DIR = "../DataLumenDS/Dataset/IRMAS_Training_Data/"
    instruments = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

    for instrument in instruments:
        
        save_path = os.path.join(SAVE_DIR, instrument)
        data_path = os.path.join(DATA_DIR, instrument)

        for root, _, files in os.walk(data_path):
            for file in files:
                cnter+=1
                file_path = os.path.join(root, file)

                # Save spectrograms
                save_spectogram(file_path, save_path, instrument)
                print(cnter)

    # X_test = np.array(X_test)
    # y_test = np.array(y_test)
    # np.save("test.npy", X_test)
    # np.save("test_labels.npy", y_test)
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # np.save("train.npy", X_train)
    # np.save("train_labels.npy", y_train)
        



