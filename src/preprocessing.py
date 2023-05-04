import librosa
import numpy as np
import os
import config

class ProcessedAudio:
    def __init__(self, file_path, features, augmentation=False):
        self.file_path = file_path
        self.features = features
        self.augmentation = augmentation

    def get_num_missing_samples(self, y):
        return (config.SAMPLE_RATE * config.DURATION) - len(y) % (config.SAMPLE_RATE * config.DURATION)

    def split_file(self):
        # Load data
        y = librosa.load(self.file_path, sr=config.SAMPLE_RATE, mono=True)[0]

        num_missing_samples = self.get_num_missing_samples(y)

        if num_missing_samples != (config.SAMPLE_RATE * config.DURATION):
            y = np.pad(y, (num_missing_samples, 0), 'constant')

        # Split data
        # frame_length is the number of samples in each frame
        # hop_length is the number of samples between the starts of consecutive frames
        frame_length = int(config.SAMPLE_RATE * config.DURATION)
        split_signals = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length) 
                 
        return split_signals
    
    def augment_data(self, signal):
        pitch_shifted_up = librosa.effects.pitch_shift(signal, sr=config.SAMPLE_RATE, n_steps=config.PITCH_SHIFT)
        pitch_shifted_down = librosa.effects.pitch_shift(signal, sr=config.SAMPLE_RATE, n_steps=-config.PITCH_SHIFT)
        y_stretched = librosa.effects.time_stretch(signal, rate=config.STRETCH_FACTOR)
        y_stretched = np.pad(y_stretched, (self.get_num_missing_samples(y_stretched), 0), 'constant')
    
        return np.stack((pitch_shifted_up, pitch_shifted_down, y_stretched), axis=1)
    
    def get_split_signals(self):
        split_signals = self.split_file()
        if self.augmentation:
            augmented = np.empty((split_signals.shape[0], 0))
            for signal in split_signals.T: 
                temp = self.augment_data(signal)
                augmented = np.concatenate((augmented, temp), axis=1)

            split_signals = np.concatenate((split_signals, augmented), axis=1)
        
        return split_signals
        
    
class LogMelSpectrogram:
    def __init__(self, n_mels=config.N_MELS, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def get_feature(self, signal):
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=signal, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)

        # Compress with natural logarithm
        log_mel_spec = librosa.amplitude_to_db(np.abs(mel_spec))

        normalized = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())

        return normalized
    
class MFCC:
    def __init__(self, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def get_feature(self, signal):
        mfcc = librosa.feature.mfcc(y=signal, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        normalized = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())
        
        return normalized
    


class PreProcessingPipeline:
    def __init__(self, feature, augmentation, save_path, data_path):
        self.feature = feature
        self.augmentation = augmentation
        self.save_path = save_path
        self.data_path = data_path
    
    def save_preprocessed_data(self, data, save_path, file, i):
        save_path = os.path.join(save_path, file + f"_{i}.npy")
        np.save(save_path, data)

    def run(self):
        for instrument in config.INSTRUMENTS:
        
            save_path = os.path.join(config.PREPROCESSED_DATA_PATH, instrument)
            data_path = os.path.join(config.DATA_PATH, instrument)

            for root, _, files in os.walk(data_path):
                for file in files:
                    file_path = os.path.join(root, file)

                    processed_audio = ProcessedAudio(file_path, self.feature, self.augmentation)
                    split_signals = processed_audio.get_split_signals()
                    
                    for i, signal in enumerate(split_signals.T):
                        feature = self.feature.get_feature(signal)
                        self.save_preprocessed_data(feature, save_path, file, i)
                        print(f"Saved {file} to {save_path}")
                    


if __name__ == "__main__":
    preprocessor = PreProcessingPipeline(feature=LogMelSpectrogram(), augmentation=True, save_path=config.PREPROCESSED_DATA_PATH, data_path=config.DATA_PATH)
    preprocessor.run()
