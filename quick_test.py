#!/usr/bin/env python3
"""Quick test to verify NPU inference works (detects hang-on-second-inference bug)."""

import sys
import os
import signal

# Timeout handler - if NPU hangs, exit with error
def timeout_handler(signum, frame):
    print("TIMEOUT: NPU appears to have hung during inference", file=sys.stderr)
    sys.exit(2)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

# Set up environment
os.environ.setdefault('AIPULIB_PATH', '/usr/share/cix/lib/onnxruntime')
os.environ.setdefault('OPERATOR_PATH', '/usr/share/cix/lib/onnxruntime/operator')
os.environ.setdefault('GRAPH_PATH', '/tmp/zhouyi_cache/graph')
os.environ.setdefault('INTERMIDIATE_PATH', '/tmp/zhouyi_cache/intermediate')
os.makedirs('/tmp/zhouyi_cache/graph', exist_ok=True)
os.makedirs('/tmp/zhouyi_cache/intermediate', exist_ok=True)

try:
    import numpy as np
    import onnxruntime as ort

    model_path = "/home/orangepi/onnx/mnist-12-int8.onnx"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    providers = ort.get_available_providers()
    if "ZhouyiExecutionProvider" not in providers:
        print(f"WARNING: ZhouyiExecutionProvider not available. Available: {providers}",
              file=sys.stderr)

    sess = ort.InferenceSession(model_path, providers=["ZhouyiExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 1, 28, 28).astype(np.float32)

    errors = 0
    warnings = 0
    for i in range(5):
        try:
            result = sess.run(None, {input_name: dummy})
        except Exception as e:
            errors += 1
            print(f"Inference {i+1} failed: {e}", file=sys.stderr)

    signal.alarm(0)  # Cancel timeout
    print(f"Total errors: {errors},  warnings: {warnings}")
    sys.exit(0 if errors == 0 else 1)

except Exception as e:
    signal.alarm(0)
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
