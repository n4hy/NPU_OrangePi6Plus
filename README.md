# CIX Zhouyi NPU Driver Fix for Orange Pi 6 Plus

**TL;DR:** The stock CIX NPU driver hangs on the second inference. This repo contains a kernel patch that fixes it, plus benchmarks proving stability.

**Validated on Ubuntu 24.04.3 *and* 24.04.4 LTS.** On 24.04.4 the point-release upgrade removes the kernel headers/build tree, so DKMS can no longer rebuild the module — see [Ubuntu 24.04.4 note](#ubuntu-24044-note-kernel-headers) before installing. The C patch itself is unchanged between the two releases.

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
| OS | **Ubuntu 24.04.3 LTS** (original) / **24.04.4 LTS** (re-validated 2026-07-06) |
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

**On Ubuntu 24.04.4**, if `install_npu_fix_v4.sh` fails at the DKMS build step (empty
`build/` dir, no `make.log`), the kernel headers were removed by the point-release
upgrade. Run the header-restore + rebuild helper, then you're done:

```bash
sudo ./finish_npu_fix.sh
```

See the [Ubuntu 24.04.4 note](#ubuntu-24044-note-kernel-headers) below for details.

### Verify It Works

```bash
~/.pyenv/versions/cix-ort311/bin/python quick_test.py
```

Expected:
```
Total errors: 0,  warnings: 0
```
Exit code `0`. Before the fix, the second inference returns `Error code: 50`
(`poll /dev/aipu [fail]`, *"AIPU device state is abnormal"*) and hangs until timeout.

### Run Full Benchmarks

```bash
~/.pyenv/versions/cix-ort311/bin/python benchmark_npu.py
```

---

## Ubuntu 24.04.4 note (kernel headers)

The C patch is **identical** across 24.04.3 and 24.04.4 — the two string anchors in
`install_npu_fix_v4.sh` still match the stock `aipu_job_manager.c` byte-for-byte, and
the patch applies cleanly. What changes on 24.04.4 is the **build environment**, not the fix.

The 24.04.3 → 24.04.4 point-release upgrade:

1. **Reinstalls the stock driver source**, wiping any applied patch and the `.orig` backup.
   The second-inference hang returns until you re-apply the fix.
2. **Removes the kernel headers / build tree.** `/lib/modules/6.6.89-cix/build` and
   `/usr/src/linux-headers-6.6.89-cix` disappear, so `dkms build` fails immediately with
   an empty `build/` directory and no `make.log`. DKMS status is stuck at `added` and the
   patched module is never produced.

`finish_npu_fix.sh` handles both — it restores the headers from the on-image package
(`/opt/linux-headers-next-cix_1.0.2_arm64.deb`, falling back to `/mnt/tmp/...`), recreates
the `/lib/modules/6.6.89-cix/build` symlink, then rebuilds, installs, and reloads the
patched module. After it runs:

```
$ dkms status | grep aipu
aipu/5.11.0, 6.6.89-cix, aarch64: installed
```

If you are on a fresh 24.04.4 image, run `sudo ./install_npu_fix_v4.sh` (applies the patch)
followed by `sudo ./finish_npu_fix.sh` (restores headers + builds). On 24.04.3 the headers
are already present and `install_npu_fix_v4.sh` alone is sufficient.

---

## Benchmark Results

Originally tested 2026-02-04 on Ubuntu 24.04.3; re-validated 2026-07-06 on Ubuntu 24.04.4
(kernel 6.6.89-cix) after restoring the headers with `finish_npu_fix.sh`.

### Stability

Original run (24.04.3, 2026-02-04):

| Test | Inferences | Errors | Duration |
|------|------------|--------|----------|
| Quick stability | 1,000 | 0 | 0.34s |
| Sustained load | 136,618 | 0 | 60s |
| Session recreation | 100 (50 sessions × 2) | 0 | 4.4s |

Re-validation run (24.04.4, 2026-07-06, patched module rebuilt via `finish_npu_fix.sh`):

| Test | Inferences | Errors | Duration | Rate |
|------|------------|--------|----------|------|
| `quick_test.py` | 5 | 0 | <1s | exit 0 |
| Stability (benchmark Test 1) | 1,000 | 0 | 0.36s | 2,808 inf/sec |

**The fix works.** Before the patch, inference #2 hangs 100% of the time with
`Error code: 50` / `poll /dev/aipu [fail]`. After the patch, 1,000 consecutive
inferences complete with zero failures.

### NPU Performance Summary (24.04.4, 2026-07-06)

| Model | Size | MACs | Latency (mean) | Throughput | TOPS |
|-------|------|------|----------------|------------|------|
| MNIST INT8 | 11KB | 322K | 0.322 ms | 3,101/sec | 0.002 |
| **MobileNetV2 INT8** | 3.5MB | 300M | **1.80 ms** | **555/sec** | **0.33** |

The **MobileNetV2 result (~0.33 TOPS)** is a realistic measure of NPU compute
performance. (The original 24.04.3 run measured 1.43 ms / 700 inf/sec / 0.42 TOPS;
run-to-run variation is expected.)

### Latency Details (500 timed iterations, 50 warmup, single session per process)

| Metric | MNIST INT8 | MobileNetV2 INT8 |
|--------|------------|------------------|
| Mean | 0.322 ms | 1.80 ms |
| P50 | 0.306 ms | 1.73 ms |
| P95 | 0.389 ms | 2.36 ms |
| P99 | 0.659 ms | 3.83 ms |
| Min | 0.280 ms | 1.30 ms |
| Max | 1.010 ms | 7.41 ms |
| Throughput | 3,101 inf/sec | 555 inf/sec |

> **benchmark_npu.py note:** on this 24.04.4 / onnxruntime-zhouyi 1.20.0 stack, creating a
> **second** `InferenceSession` in the same process raises
> `TypeError: 'builtin_function_or_method' object does not support vectorcall`. This is a
> userspace binding quirk, **not** the kernel hang — the stability test (single session,
> 1,000 inferences) passes. The latency numbers above were collected with one session per
> process to work around it.

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

Expected output (24.04.3 shown; 24.04.4 reports `VERSION="24.04.4 LTS (Noble Numbat)"` — both work):
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

Claude was invaluable in sorting out the last stages. It wrote ALL the tests that identified the order of operations needed to make this work.

