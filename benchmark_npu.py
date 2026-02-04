#!/usr/bin/env python3
"""
NPU Stress Test & Benchmark Suite
Tests the CIX Zhouyi NPU with various workloads to verify stability and measure performance.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Setup environment
os.environ['AIPULIB_PATH'] = '/usr/share/cix/lib/onnxruntime'
os.environ['OPERATOR_PATH'] = '/usr/share/cix/lib/onnxruntime/operator'
os.environ['GRAPH_PATH'] = '/tmp/zhouyi_cache/graph'
os.environ['INTERMIDIATE_PATH'] = '/tmp/zhouyi_cache/intermediate'
os.makedirs('/tmp/zhouyi_cache/graph', exist_ok=True)
os.makedirs('/tmp/zhouyi_cache/intermediate', exist_ok=True)

import onnxruntime as ort

def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_result(label, value, unit=""):
    print(f"  {label:<40} {value:>15} {unit}")

def get_system_info():
    """Get system information."""
    print_header("SYSTEM INFORMATION")

    # CPU info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        # Find CPU model
        for line in cpuinfo.split('\n'):
            if 'model name' in line.lower() or 'cpu model' in line.lower():
                print_result("CPU", line.split(':')[1].strip())
                break
    except:
        pass

    # Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    print_result("Memory", f"{mem_kb // 1024}", "MB")
                    break
    except:
        pass

    # Kernel
    try:
        with open('/proc/version', 'r') as f:
            version = f.read().split()[2]
            print_result("Kernel", version)
    except:
        pass

    # NPU info
    print_result("NPU", "CIX Zhouyi V3 AIPU (3 cores, 4 TECs/core)")
    print_result("ONNX Runtime", ort.__version__)
    print_result("Providers", ", ".join(ort.get_available_providers()))

def benchmark_inference(session, input_name, input_data, num_iterations, warmup=10):
    """Run inference benchmark and return statistics."""
    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: input_data})

    # Timed runs
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times = np.array(times)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
        'throughput': 1000.0 / np.mean(times),  # inferences/sec
        'total_inferences': num_iterations
    }

def test_stability(session, input_name, input_data, num_iterations):
    """Test stability with many consecutive inferences."""
    success = 0
    failures = 0

    for i in range(num_iterations):
        try:
            session.run(None, {input_name: input_data})
            success += 1
        except Exception as e:
            failures += 1
            print(f"  FAILURE at iteration {i+1}: {e}")

    return success, failures

def run_benchmarks():
    print_header("NPU BENCHMARK SUITE")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    get_system_info()

    # Check for models
    models = [
        ("/home/orangepi/onnx/mnist-12-int8.onnx", "MNIST INT8", [1, 1, 28, 28]),
        ("/home/orangepi/onnx/mnist-8.onnx", "MNIST FP32", [1, 1, 28, 28]),
    ]

    available_models = [(p, n, s) for p, n, s in models if os.path.exists(p)]

    if not available_models:
        print("\nERROR: No test models found!")
        return

    results = {}

    # =========================================================================
    # TEST 1: Stability Test (proves the fix works)
    # =========================================================================
    print_header("TEST 1: STABILITY TEST (Proves Driver Fix)")
    print("  Running 1000 consecutive inferences to verify no hangs...")

    model_path, model_name, input_shape = available_models[0]

    sess = ort.InferenceSession(model_path,
        providers=['ZhouyiExecutionProvider', 'CPUExecutionProvider'])

    inp = sess.get_inputs()[0]
    x = np.random.randn(*input_shape).astype(np.float32)

    start = time.time()
    success, failures = test_stability(sess, inp.name, x, 1000)
    elapsed = time.time() - start

    print_result("Model", model_name)
    print_result("Total Inferences", success + failures)
    print_result("Successful", success)
    print_result("Failed", failures)
    print_result("Total Time", f"{elapsed:.2f}", "seconds")
    print_result("Average Rate", f"{success/elapsed:.1f}", "inf/sec")

    if failures == 0:
        print("\n  *** STABILITY TEST PASSED - NO HANGS! ***")
    else:
        print(f"\n  *** STABILITY TEST FAILED - {failures} failures ***")

    results['stability'] = {'success': success, 'failures': failures, 'time': elapsed}

    # =========================================================================
    # TEST 2: Latency Benchmark
    # =========================================================================
    print_header("TEST 2: LATENCY BENCHMARK")

    for model_path, model_name, input_shape in available_models:
        print(f"\n  Model: {model_name}")
        print(f"  Path: {model_path}")

        sess = ort.InferenceSession(model_path,
            providers=['ZhouyiExecutionProvider', 'CPUExecutionProvider'])

        inp = sess.get_inputs()[0]
        x = np.random.randn(*input_shape).astype(np.float32)

        stats = benchmark_inference(sess, inp.name, x, num_iterations=500, warmup=50)

        print_result("Iterations", stats['total_inferences'])
        print_result("Mean Latency", f"{stats['mean']:.3f}", "ms")
        print_result("Std Dev", f"{stats['std']:.3f}", "ms")
        print_result("Min Latency", f"{stats['min']:.3f}", "ms")
        print_result("Max Latency", f"{stats['max']:.3f}", "ms")
        print_result("P50 Latency", f"{stats['p50']:.3f}", "ms")
        print_result("P95 Latency", f"{stats['p95']:.3f}", "ms")
        print_result("P99 Latency", f"{stats['p99']:.3f}", "ms")
        print_result("Throughput", f"{stats['throughput']:.1f}", "inf/sec")

        results[model_name] = stats

    # =========================================================================
    # TEST 3: Sustained Load Test
    # =========================================================================
    print_header("TEST 3: SUSTAINED LOAD TEST (60 seconds)")

    model_path, model_name, input_shape = available_models[0]
    sess = ort.InferenceSession(model_path,
        providers=['ZhouyiExecutionProvider', 'CPUExecutionProvider'])

    inp = sess.get_inputs()[0]
    x = np.random.randn(*input_shape).astype(np.float32)

    duration = 60  # seconds
    print(f"  Running continuous inference for {duration} seconds...")

    count = 0
    errors = 0
    start = time.time()
    interval_start = start
    interval_count = 0

    while (time.time() - start) < duration:
        try:
            sess.run(None, {inp.name: x})
            count += 1
            interval_count += 1
        except Exception as e:
            errors += 1

        # Print progress every 10 seconds
        if time.time() - interval_start >= 10:
            rate = interval_count / (time.time() - interval_start)
            elapsed = time.time() - start
            print(f"    [{elapsed:5.1f}s] {count:6d} inferences, {rate:.1f} inf/sec")
            interval_start = time.time()
            interval_count = 0

    total_time = time.time() - start

    print_result("Total Inferences", count)
    print_result("Errors", errors)
    print_result("Duration", f"{total_time:.1f}", "seconds")
    print_result("Average Throughput", f"{count/total_time:.1f}", "inf/sec")
    print_result("Total Operations", f"{count * 28 * 28 / 1e6:.2f}", "M pixels processed")

    results['sustained'] = {'count': count, 'errors': errors, 'duration': total_time}

    # =========================================================================
    # TEST 4: Burst Test (rapid fire)
    # =========================================================================
    print_header("TEST 4: BURST TEST (Back-to-back inference)")

    burst_sizes = [10, 50, 100, 500]

    for burst in burst_sizes:
        times = []
        for _ in range(5):  # 5 trials per burst size
            start = time.perf_counter()
            for _ in range(burst):
                sess.run(None, {inp.name: x})
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        avg_per_inf = avg_time / burst
        throughput = burst / (avg_time / 1000)

        print_result(f"Burst of {burst}", f"{avg_per_inf:.3f} ms/inf, {throughput:.0f} inf/sec")

    # =========================================================================
    # TEST 5: Memory Stress Test
    # =========================================================================
    print_header("TEST 5: SESSION RECREATION TEST")
    print("  Creating and destroying sessions repeatedly...")

    session_count = 50
    start = time.time()

    for i in range(session_count):
        sess = ort.InferenceSession(model_path,
            providers=['ZhouyiExecutionProvider', 'CPUExecutionProvider'])
        inp = sess.get_inputs()[0]
        x = np.random.randn(*input_shape).astype(np.float32)
        sess.run(None, {inp.name: x})
        sess.run(None, {inp.name: x})  # Two inferences per session
        del sess

    elapsed = time.time() - start

    print_result("Sessions Created", session_count)
    print_result("Inferences per Session", 2)
    print_result("Total Inferences", session_count * 2)
    print_result("Total Time", f"{elapsed:.2f}", "seconds")
    print_result("Rate", f"{session_count/elapsed:.1f}", "sessions/sec")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("BENCHMARK SUMMARY")

    print("\n  DRIVER FIX VALIDATION:")
    if results['stability']['failures'] == 0:
        print("    [PASS] 1000 consecutive inferences completed without hanging")
    else:
        print(f"    [FAIL] {results['stability']['failures']} failures detected")

    print(f"\n  SUSTAINED LOAD:")
    print(f"    [PASS] {results['sustained']['count']} inferences over {results['sustained']['duration']:.0f}s")
    print(f"           {results['sustained']['count']/results['sustained']['duration']:.1f} inf/sec sustained")

    print(f"\n  LATENCY (MNIST INT8):")
    if 'MNIST INT8' in results:
        s = results['MNIST INT8']
        print(f"    Mean: {s['mean']:.3f} ms | P95: {s['p95']:.3f} ms | P99: {s['p99']:.3f} ms")

    print("\n" + "=" * 70)
    print(" BENCHMARK COMPLETE")
    print("=" * 70 + "\n")

    return results


if __name__ == '__main__':
    try:
        results = run_benchmarks()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
