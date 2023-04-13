# import torch

# model = torch.jit.load('model.pt')

# dummy_input = torch.randn(1, 1, 128, 44)
# torch.onnx.export(model, dummy_input, "model.onnx")

import onnxruntime as onnxrt
import numpy as np

spec = np.load("../../../DataLumenDS/Processed/cel/008__[cel][nod][cla]0058__1.wav_0.npy")

spec = spec.reshape(1, 1, 128, 44)

onnx_session= onnxrt.InferenceSession("model.onnx")
onnx_session.set_providers(['CPUExecutionProvider'], [{'precision': 'float32'}])
onnx_inputs= {onnx_session.get_inputs()[0].name: spec}
onnx_output = onnx_session.run(None, onnx_inputs)

print(onnx_output[0][0])