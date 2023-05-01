import os
from typing import List, Tuple

import numpy as np
import onnxruntime as onnxrt
from torch import from_numpy

from preprocessingivara import create_spectogram, split_file


class ModelLoader:
    def __init__(self, model_path, provider = 'CPUExecutionProvider', config: dict = None):
        self.model = onnxrt.InferenceSession(model_path)
        if config:
            self.model.set_providers([provider], [config])
        else:
            self.model.set_providers([provider])

    def predict(self, data):
        onnx_inputs = {self.model.get_inputs()[0].name: np.expand_dims(data, 0)}
        prediction = self.model.run(None, onnx_inputs)
        return prediction[0]


class InstrumentClassifier:
    def __init__(self, tresholds, instrument_list):
        self.tresholds = tresholds
        self.instrument_list = instrument_list

    def classify(self, predictions):
        pred_max = [0] * len(self.tresholds)
        for prediction in predictions:
            for i, p in enumerate(prediction):
                if p > pred_max[i]:
                    pred_max[i] = p

        for i, p in enumerate(pred_max):
            if p > self.tresholds[i]:
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
        return self.TP / (self.TP + self.FN)

    def accuracy(self):
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
        return self.exacts / self.total


def load_data(file_path, instrument_list):
    instruments = [0] * len(instrument_list)
    with open(file_path, "r") as f:
        data = f.readlines()
        for instrument in data:
            if instrument.strip() in instrument_list:
                instruments[instrument_list.index(instrument.strip())] = 1

    file_path = file_path[:-4] + ".wav"
    # Split data
    split_signals = split_file(file_path)

    # Create spectrograms
    spectrograms = []
    for signal in split_signals.T:
        spectrogram = create_spectogram(signal)
        spectrograms.append(np.expand_dims(spectrogram, 2))

    return instruments, spectrograms

if __name__ == "__main__":
    # Load model
    model_path = "path/to/model.onnx"
    model_loader = ModelLoader(model_path)

    # Load data
    path = '../../DataLumenDS/Dataset/IRMAS_Validation_Data/'
    tresholds = [0.85, 0.85, 0.8, 0.5, 0.7, 0.7, 0.7, 0.85, 0.85, 0.8, 0.7]
    metrics = EvaluationMetrics()
    instrument_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path[-3:] == "txt":
                # Load data
                instruments, spectrograms = load_data(file_path, instrument_list)

                # Predict
                predictions = model_loader.predict(np.array(spectrograms))
                pred_max = InstrumentClassifier.classify(predictions)

                # Evaluate
                metrics.evaluate(pred_max, instruments)

    # Print results
    print("Accuracy: ", metrics.accuracy())
    print("Recall: ", metrics.recall())
    print("Precision: ", metrics.precision())
    print("F1 score: ", metrics.f1_score())
    print("Exact matches: ", metrics.exact_matches())

