import librosa
import numpy as np
import os
import config


instruments = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]


class PreProcessingPipeline:
    def __init__(self):
        pass
        





def split_file(data):
    """ Split audio file into samples of length config.DURATION seconds """
    # Load data
    y = librosa.load(data, sr=config.SAMPLE_RATE, mono=True)[0]


    num_missing_samples = (config.SAMPLE_RATE * config.DURATION) - len(y) % (config.SAMPLE_RATE * config.DURATION)

    if num_missing_samples == (config.SAMPLE_RATE * config.DURATION):
        num_missing_samples = 0

    padded_signal = np.pad(y, (num_missing_samples, 0), 'constant')
   
    # Split data
    # frame_length is the number of samples in each frame
    # hop_length is the number of samples between the starts of consecutive frames
    frame_length = int(config.SAMPLE_RATE * config.DURATION)
    split_signals = librosa.util.frame(padded_signal, frame_length=frame_length, hop_length=frame_length) 
                 
    return split_signals

def create_spectogram(signal):
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, n_mels=config.N_MELS, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)

    # Compress with natural logarithm
    log_mel_spec = librosa.amplitude_to_db(np.abs(mel_spec))

    normalized = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())


    return normalized

def create_mfcc(signal):
    pass


noise_factor = 0.08
def augment_data(signal):
    noise = np.random.randn(len(signal))
    noised_signal = signal + noise_factor * noise

    pitch_shifted = librosa.effects.pitch_shift(signal, sr=config.SAMPLE_RATE, n_steps=2)

    return [noised_signal, pitch_shifted]

cnt = 0
def save_spectogram(data, path, instrument):
    global cnt
    """ Save spectrogram of each sample of the audio file """
    # Split data
    cnt += 1
    split_signals = split_file(data)
    tempy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tempy[instruments.index(instrument)] = 1
    
    for i, signal in enumerate(split_signals.T):
        log_spectrogram = create_spectogram(signal)
        file_name = os.path.split(data)[1]
        save_path = os.path.join(path, file_name + f"_{i}.npy")

        np.save(save_path, log_spectrogram)
        
   



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



