import os
import numpy as np
import onnxruntime as onnxrt
from preprocessing import ProcessedAudio, LogMelSpectrogram
import config


class ModelLoader:
    def __init__(self, model_path, provider = 'CPUExecutionProvider'):
        self.model = onnxrt.InferenceSession(model_path)
        self.model.set_providers([provider])

    def predict(self, data):
        onnx_inputs = {self.model.get_inputs()[0].name: np.expand_dims(data, 0)}
        prediction = self.model.run(None, onnx_inputs)
        return prediction[0]


class InstrumentClassifier:
    def __init__(self, instrument_list):
        self.instrument_list = instrument_list

    def classify(self, spectrograms, model):
        pred_max = [0] * len(self.instrument_list)
        for spectrogram in spectrograms:
            prediction = model.predict(np.resize(spectrogram, (1, 128, 44)))
            for i, pred in enumerate(prediction[0]):
                if pred > pred_max[i]:
                    pred_max[i] = pred

        for i, pred in enumerate(pred_max):
            if pred > config.THRESHOLD:
                pred_max[i] = 1
            else:
                pred_max[i] = 0
    
        return pred_max


class EvaluationMetrics:
    def __init__(self):
        self.total = 0
        self.exacts = 0
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def evaluate(self, pred_max, instruments):
        self.total += 1
        if pred_max == instruments:
            self.exacts += 1

        for i, pred in enumerate(pred_max):
            if pred == instruments[i]:
                if pred == 1:
                    self.TP += 1
                else:
                    self.TN += 1
            else:
                if pred == 1:
                    self.FP += 1
                else:
                    self.FN += 1

    def recall(self):
        return self.TP / (self.TP + self.FN) if self.TP + self.FN != 0 else 0

    def hamming(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def precision(self):
        if self.TP + self.FP == 0:
            return 0
        else:
            return self.TP / (self.TP + self.FP)

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)

    def exact_matches(self):
        return self.exacts / self.total if self.total != 0 else 0


def load_data(file_path, instrument_list):
    instruments = [0] * len(instrument_list)
    with open(file_path, "r") as f:
        data = f.readlines()
        for instrument in data:
            if instrument.strip() in instrument_list:
                instruments[instrument_list.index(instrument.strip())] = 1

    file_path = file_path[:-4] + ".wav"
    # Split data
    logmel = LogMelSpectrogram()
    audio = ProcessedAudio(file_path, logmel, "")
    split_signals = audio.split_file()

    # Create spectrograms
    spectrograms = []
    for signal in split_signals.T:
        spectrogram = logmel.get_feature(signal)
        spectrograms.append(np.expand_dims(spectrogram, 2))

    return instruments, spectrograms

if __name__ == "__main__":
    # Load model
    model_path = "./model/models/model_densenet_121.onnx"
    model_loader = ModelLoader(model_path)

    # Load data
    path = config.VALIDATION_DATA_PATH
    total_metrics = EvaluationMetrics()
    instrument_metrics = {instrument: EvaluationMetrics() for instrument in config.INSTRUMENTS}

    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path[-3:] == "txt":
                # Load data
                instruments, spectrograms = load_data(file_path, config.INSTRUMENTS)

                # Predict
                classifier = InstrumentClassifier(config.INSTRUMENTS)
                pred_max = classifier.classify(spectrograms, model_loader)

                # Evaluate
                total_metrics.evaluate(pred_max, instruments)

                for i, instrument in enumerate(config.INSTRUMENTS):
                    instrument_metrics[instrument].evaluate([pred_max[i]], [instruments[i]])


                # Print results
                print("Total:")
                print("Hamming score: ", total_metrics.hamming())
                print("Recall: ", total_metrics.recall())
                print("Precision: ", total_metrics.precision())
                print("F1 score: ", total_metrics.f1_score())
                print("Exact matches: ", total_metrics.exact_matches())
                print("--------------------------------------")
        
    
                for instrument in instrument_metrics.keys():
                    print(f'{instrument}:')
                    print("Hamming score: ", instrument_metrics[instrument].hamming())
                    print("Recall: ", instrument_metrics[instrument].recall())
                    print("Precision: ", instrument_metrics[instrument].precision())
                    print("F1 score: ", instrument_metrics[instrument].f1_score())
                    print("Exact matches: ", instrument_metrics[instrument].exact_matches())
                    print("Occurences: ", instrument_metrics[instrument].TP + instrument_metrics[instrument].FN)
                    print("--------------------------------------")