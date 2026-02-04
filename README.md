# CIX Zhouyi NPU Driver Fix for Orange Pi 6 Plus

**TL;DR:** The stock CIX NPU driver hangs on the second inference. This repo contains a kernel patch that fixes it, plus benchmarks proving stability.

---

## Tested Environment (EXACT versions)

**This fix was developed and tested with these EXACT versions. Other versions may or may not work.**

### Hardware

| Component | Value |
|-----------|-------|
| Board | **Orange Pi 6 Plus** |
| SoC | CIX CD8180 (SKY1) / CIX P1 CD8160 |
| CPU | ARM Cortex-X4 (0xd81) |
| NPU | Zhouyi V3 AIPU (ARM China) |
| NPU Config | 3 cores, 4 TECs per core |

### Operating System

| Component | Version |
|-----------|---------|
| OS | **Ubuntu 24.04.3 LTS (Noble Numbat)** |
| Orange Pi Image | **1.0.2 Noble** |
| Image Type | user-built |
| Board Family | cix |
| Architecture | arm64 (aarch64) |

### Kernel

| Component | Version |
|-----------|---------|
| Kernel | **6.6.89-cix** |
| Kernel Build | `#90 SMP PREEMPT Tue Dec 30 20:43:11 CST 2025` |
| Linux Family | cix |
| Branch | next |

### NPU Driver (Kernel Module)

| Component | Version |
|-----------|---------|
| DKMS Package | **aipu 5.11.0** |
| Module File | `/lib/modules/6.6.89-cix/updates/dkms/aipu.ko` |
| Source Location | `/usr/src/aipu-5.11.0/armchina-npu/` |
| Patched File | `aipu_job_manager.c` |
| Original File SHA256 | `1841a5f691abeb6f89d16a264e4044701c33027b2a50e690d5809f8617a2a751` |

### System Packages (apt)

| Package | Version | Description |
|---------|---------|-------------|
| **cix-npu-driver** | **1.0.0+2503.orangepi** | Kernel driver source (DKMS) |
| **cix-noe-umd** | **2.0.4** | User-mode driver |
| **cix-npu-onnxruntime** | **1.1.0** | ONNX Runtime with Zhouyi EP |

### Python Environment

| Component | Version |
|-----------|---------|
| Python | **3.11.7** |
| Environment | pyenv virtualenv `cix-ort311` |
| numpy | **1.26.4** |

### Python Packages (pip)

| Package | Version | Description |
|---------|---------|-------------|
| **onnxruntime-zhouyi** | **1.20.0** | ONNX Runtime with Zhouyi Execution Provider |
| **libnoe** | **2.0.1** | NPU runtime library |
| **noe_engine** | **2.0.1** | NPU inference engine |
| **ZhouyiOperators-x2** | **25.4.23** | NPU operator library |

### Runtime Libraries

| File | SHA256 (first 16 chars) |
|------|-------------------------|
| `/usr/share/cix/lib/onnxruntime/libonnxruntime.so` | `55acd5881dcbb86e...` |

### Library Dates (from filesystem)

```
Sep 30 08:34 libaipu_buildingtool.so
Sep 30 08:34 libaipu_driver.so
Sep 30 08:34 libonnxruntime.so
```

---

## The Problem

The stock CIX NPU driver (`aipu.ko` v5.11.0) shipped with Orange Pi 6 Plus has a **critical race condition** that causes the second NPU inference to hang indefinitely. Every user of this board's NPU is affected.

### Root Cause

A race condition exists between userspace job submission and the kernel's interrupt bottom half:

```
1. Job 1 completes → IRQ upper half marks job as SUCCESS
2. Userspace polls, sees job 1 complete, submits job 2
3. schedule_v3_job_no_lock() sees pool->created=true, links job 2 to old TCB chain
4. IRQ bottom half finally runs, destroys command pool (too late!)
5. Job 2 is linked to invalid TCB → hardware never fires completion → HANGS FOREVER
```

### The Fix

In `schedule_v3_job_no_lock()`, detect when all previous jobs have completed but the hardware command pool still exists (stale state). Destroy the stale pool and create a fresh one.

**File:** `/usr/src/aipu-5.11.0/armchina-npu/aipu_job_manager.c`

---

## Quick Start

### Prerequisites

You MUST be running the Orange Pi official Ubuntu 24.04 image with the pre-installed CIX NPU SDK. The SDK is not available for download separately.

### Install the Fix

```bash
git clone https://github.com/n4hy/NPU_OrangePi6Plus.git
cd NPU_OrangePi6Plus
sudo ./install_npu_fix_v4.sh
```

### Verify It Works

```bash
~/.pyenv/versions/cix-ort311/bin/python quick_test.py
cat /tmp/npu_test_output.txt
```

Expected:
```
Step 7: Running inference 2...
Step 8: Inference 2 DONE! Shape: (1, 10)
=== ALL TESTS PASSED ===
```

### Run Full Benchmarks

```bash
~/.pyenv/versions/cix-ort311/bin/python benchmark_npu.py
```

---

## Benchmark Results

Tested on 2026-02-04 with driver fix v4.

### Stability: 136,618 Consecutive Inferences, Zero Failures

| Test | Inferences | Errors | Duration |
|------|------------|--------|----------|
| Quick stability | 1,000 | 0 | 0.34s |
| Sustained load | 136,618 | 0 | 60s |
| Session recreation | 100 (50 sessions × 2) | 0 | 4.4s |

**The fix works.** Before the patch, inference #2 would hang 100% of the time.

### NPU Performance Summary

| Model | Size | MACs | Latency | Throughput | TOPS |
|-------|------|------|---------|------------|------|
| MNIST INT8 | 11KB | 322K | 0.35 ms | 2,900/sec | 0.002 |
| **MobileNetV2 INT8** | 3.5MB | 300M | **1.43 ms** | **700/sec** | **0.42** |

The **MobileNetV2 result (0.42 TOPS)** is a realistic measure of NPU compute performance.

### Latency Details (MNIST INT8)

| Metric | Value |
|--------|-------|
| Mean | 0.345 ms |
| P50 | 0.337 ms |
| P95 | 0.433 ms |
| P99 | 0.606 ms |

### Latency Details (MobileNetV2 INT8)

| Metric | Value |
|--------|-------|
| Mean | 1.43 ms |
| Min | 1.26 ms |
| Max | 2.54 ms |
| Throughput | 700 inf/sec |

### Throughput Comparison

| Model | INT8 | FP32 | Speedup |
|-------|------|------|---------|
| MNIST | 2,907 inf/sec | 696 inf/sec | 4.2x |
| MobileNetV2 | 700 inf/sec | - | - |

INT8 is **4x faster** than FP32. Quantize your models.

### Model Compatibility Notes

The Zhouyi NPU requires models with **fixed input dimensions**. Models with dynamic batch size (e.g., `'N'` instead of `1`) will fall back to CPU.

Working models can be downloaded from:
- [Kalray MobileNetV2 INT8](https://huggingface.co/Kalray/mobilenet-v2) - works on NPU
- [ONNX Model Zoo](https://huggingface.co/onnxmodelzoo) - check for fixed dimensions
- [ARM China Model Zoo](https://github.com/Arm-China/Model_zoo) - requires Zhouyi SDK to compile

---

## Technical Details

### What the Patch Does

Before deciding how to schedule a new job, check if there are any RUNNING jobs. If `pool->created=true` but no jobs are RUNNING, the pool is stale:

1. Call `partition->ops->destroy_command_pool()` to destroy hardware state
2. Clear `pool->qlist` arrays with `memset()`
3. Reset `pool->created = false`
4. Reset `manager->tec_intr_en = false`
5. Call `aipu_mm_set_final_htbuf_index(manager->mm, -1)`
6. Proceed with `ZHOUYI_TRIGGER_TYPE_CREATE`

### Patch Location

```c
// In schedule_v3_job_no_lock(), around line 457
// Replace the trigger_type decision logic
```

See `install_npu_fix_v4.sh` for the complete patch.

### Revert to Stock Driver

```bash
sudo cp /usr/src/aipu-5.11.0/armchina-npu/aipu_job_manager.c.orig \
        /usr/src/aipu-5.11.0/armchina-npu/aipu_job_manager.c
sudo dkms remove aipu/5.11.0 --all
sudo dkms add /usr/src/aipu-5.11.0
sudo dkms build aipu/5.11.0
sudo dkms install aipu/5.11.0
sudo rmmod aipu && sudo modprobe aipu
```

---

## Environment Setup

### Required Environment Variables

```bash
export AIPULIB_PATH=/usr/share/cix/lib/onnxruntime
export OPERATOR_PATH=/usr/share/cix/lib/onnxruntime/operator
export GRAPH_PATH=/tmp/zhouyi_cache/graph
export INTERMIDIATE_PATH=/tmp/zhouyi_cache/intermediate
export LD_LIBRARY_PATH=/usr/share/cix/lib/onnxruntime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$GRAPH_PATH" "$INTERMIDIATE_PATH"
```

### Verify NPU is Available

```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected: ['ZhouyiExecutionProvider', 'CPUExecutionProvider']
```

### NPU Power Management

Keep NPU active (disable runtime PM suspend):

```bash
echo "on" | sudo tee /sys/devices/platform/CIXH4000:00/power/control
```

---

## Verifying Your Environment Matches

Run these commands to check your versions match:

```bash
# OS version
cat /etc/os-release | grep VERSION=

# Kernel
uname -r

# NPU packages
dpkg -l | grep -E "cix-(npu|noe)"

# Python packages (in your pyenv)
pip list | grep -iE "(onnx|noe|zhouyi|libnoe)"

# Driver source checksum (before patching)
sha256sum /usr/src/aipu-5.11.0/armchina-npu/aipu_job_manager.c
```

Expected output:
```
VERSION="24.04.3 LTS (Noble Numbat)"
6.6.89-cix
cix-noe-umd      2.0.4
cix-npu-driver   1.0.0+2503.orangepi
cix-npu-onnxruntime 1.1.0
onnxruntime-zhouyi  1.20.0
libnoe              2.0.1
noe_engine          2.0.1
ZhouyiOperators-x2  25.4.23
1841a5f691abeb6f89d16a264e4044701c33027b2a50e690d5809f8617a2a751  (original, unpatched)
```

---

## Known Limitations

1. **Model Compatibility:** Complex Transformer layers may fall back to CPU
2. **Quantization:** NPU is optimized for INT8; FP32 is 4x slower
3. **Memory:** Large models (>2-3GB) may crash the compiler
4. **SDK Availability:** NeuralONE SDK only available pre-installed on official Orange Pi image
5. **Version Dependency:** This patch targets aipu driver 5.11.0; other versions may differ

---

## Contributing

If you have access to larger models (ResNet, YOLO, etc.) quantized for Zhouyi, please run benchmarks and submit results.

If you test this on different SDK versions, please report whether it works.

---

## License

This fix is provided as-is for educational and research purposes. The original driver is proprietary to ARM China / CIX.

---

## Acknowledgments

- Bug analysis and fix developed with assistance from Claude (Anthropic)
- Orange Pi for providing the hardware
- ARM China for the Zhouyi NPU architecture
