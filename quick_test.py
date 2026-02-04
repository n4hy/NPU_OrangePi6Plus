#!/usr/bin/env python3
import os
import sys

# Redirect all output to file
log_file = open('/tmp/npu_test_output.txt', 'w', buffering=1)
sys.stdout = log_file
sys.stderr = log_file

print("=== NPU Quick Test ===", flush=True)

os.environ['AIPULIB_PATH'] = '/usr/share/cix/lib/onnxruntime'
os.environ['OPERATOR_PATH'] = '/usr/share/cix/lib/onnxruntime/operator'
os.environ['GRAPH_PATH'] = '/tmp/zhouyi_cache/graph'
os.environ['INTERMIDIATE_PATH'] = '/tmp/zhouyi_cache/intermediate'
os.makedirs('/tmp/zhouyi_cache/graph', exist_ok=True)
os.makedirs('/tmp/zhouyi_cache/intermediate', exist_ok=True)

import numpy as np
print("Step 1: Importing onnxruntime...", flush=True)
import onnxruntime as ort
print(f"Step 2: Providers: {ort.get_available_providers()}", flush=True)

print("Step 3: Creating session...", flush=True)
sess = ort.InferenceSession("/home/orangepi/onnx/mnist-12-int8.onnx",
    providers=['ZhouyiExecutionProvider', 'CPUExecutionProvider'])
print(f"Step 4: Session created. Providers: {sess.get_providers()}", flush=True)

inp = sess.get_inputs()[0]
x = np.zeros([1, 1, 28, 28], dtype=np.float32)

print("Step 5: Running inference 1...", flush=True)
r = sess.run(None, {inp.name: x})
print(f"Step 6: Inference 1 DONE! Shape: {r[0].shape}", flush=True)

print("Step 7: Running inference 2...", flush=True)
r = sess.run(None, {inp.name: x})
print(f"Step 8: Inference 2 DONE! Shape: {r[0].shape}", flush=True)

print("=== ALL TESTS PASSED ===", flush=True)
log_file.close()
