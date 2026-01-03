---
description: TensorRT optimization and deployment for YOWO on Jetson Orin Nano
---

# TensorRT Optimization for YOWO on Jetson Orin Nano

This workflow documents how to optimize, build, and deploy YOWO with TensorRT for maximum performance.

## Prerequisites

```bash
conda activate yowov2
```

---

## Step 1: Set Orin Nano to Maximum Performance Mode

// turbo-all

### Check Available Power Modes

```bash
cat /etc/nvpmodel.conf | grep -E "POWER_MODEL ID|NAME="
```

**Available modes on Orin Nano Super:**
- `0` = 15W (default, GPU max 625 MHz)
- `1` = 25W (GPU max 918 MHz)
- `2` = MAXN_SUPER (GPU max 1020 MHz) ← **USE THIS**

### Enable Maximum Performance

```bash
# Set to MAXN_SUPER mode
sudo nvpmodel -m 2

# Lock clocks at maximum frequency
sudo jetson_clocks

# Verify GPU is at max clock
cat /sys/devices/platform/gpu.0/devfreq/*/cur_freq
# Should show: 1020000000 (1020 MHz)
```

### Check Current Power Mode

```bash
nvpmodel -q
```

---

## Step 2: Export ONNX Model

### Export at Different Resolutions

```bash
# 224 resolution (fastest, ~14 FPS on Orin Nano Super)
python export_onnx.py \
    --weight yowo_v2_x3d_m_yolo11m_multitask_epoch_14.pth \
    --output yowo_multitask_224.onnx \
    --img_size 224

# 256 resolution (middle ground, ~10-11 FPS estimated)
python export_onnx.py \
    --weight yowo_v2_x3d_m_yolo11m_multitask_epoch_14.pth \
    --output yowo_multitask_256.onnx \
    --img_size 256

# 320 resolution (best quality, ~7-8 FPS)
python export_onnx.py \
    --weight yowo_v2_x3d_m_yolo11m_multitask_epoch_14.pth \
    --output yowo_multitask_320.onnx \
    --img_size 320
```

### ONNX Output Format

```
Input:  [1, 3, 16, H, W]  (batch, channels, frames, height, width)
Output: [N, 224]          (N detections, 224 values each)

Output breakdown per detection:
  [0]:       confidence (sigmoid applied)
  [1:37]:    object class probabilities (softmax, 36 classes)
  [37:194]:  action probabilities (sigmoid, 157 classes)
  [194:220]: relation probabilities (sigmoid, 26 classes)
  [220:224]: bounding box [x1, y1, x2, y2] in pixel coords
```

---

## Step 3: Build TensorRT Engine

### Basic FP16 Build

```bash
trtexec \
    --onnx=yowo_multitask_224.onnx \
    --saveEngine=yowo_multitask_224_fp16.engine \
    --fp16 \
    --memPoolSize=workspace:6144
```

### Optimized Build (Recommended)

```bash
trtexec \
    --onnx=yowo_multitask_320.onnx \
    --saveEngine=yowo_multitask_320_fp16_v2.engine \
    --fp16 \
    --memPoolSize=workspace:6144 \
    --builderOptimizationLevel=5
```

**Note:** `--builderOptimizationLevel=5` is the maximum and takes longer to build but produces the fastest engine.

### Key trtexec Flags

| Flag | Description |
|------|-------------|
| `--fp16` | Enable FP16 precision (2x faster, minimal accuracy loss) |
| `--memPoolSize=workspace:6144` | 6GB workspace for optimization |
| `--builderOptimizationLevel=5` | Maximum optimization (longer build) |
| `--useCudaGraph` | Use CUDA graphs (faster execution) |
| `--useSpinWait` | Reduce latency variance |

---

## Step 4: Benchmark Performance

### Quick Benchmark

```bash
trtexec \
    --loadEngine=yowo_multitask_320_fp16.engine \
    --iterations=50 \
    --warmUp=2000 \
    --useCudaGraph \
    --useSpinWait
```

### Detailed Benchmark

```bash
trtexec \
    --loadEngine=yowo_multitask_320_fp16.engine \
    --iterations=100 \
    --warmUp=3000 \
    --useCudaGraph \
    --useSpinWait \
    --duration=10 \
    2>&1 | grep -E "(Latency|Throughput|Performance|GPU Compute)"
```

### Pure GPU Compute (No Data Transfers)

```bash
trtexec \
    --loadEngine=yowo_multitask_320_fp16.engine \
    --iterations=100 \
    --warmUp=3000 \
    --useCudaGraph \
    --useSpinWait \
    --noDataTransfers
```

---

## Step 5: Performance Reference

### Orin Nano Super (MAXN_SUPER Mode @ 1020 MHz)

| Resolution | Mean Latency | Min Latency | Throughput | FPS |
|------------|--------------|-------------|------------|-----|
| 224 | 71 ms | 52 ms | 14.0 qps | ~14 |
| 256 | ~90 ms (est) | ~70 ms (est) | ~11 qps | ~11 |
| 320 | 140 ms | 100 ms | 7.1 qps | ~7-8 |

### Resolution Scaling Formula

```
Compute ∝ (H × W)² for 3D video models

(320/224)² = 2.04x slower
(256/224)² = 1.31x slower
```

---

## Step 6: Run Demo

### 224 Resolution Demo

```bash
python demo_web_trt_pro.py --engine yowo_multitask_fp16.engine
```

### 320 Resolution Demo

```bash
python demo_web_trt_320.py --engine yowo_multitask_320_fp16_v2.engine
```

Access at: http://<ORIN_IP>:5000

---

## Troubleshooting

### Engine Build Fails with Workspace Error

Increase workspace size:
```bash
--memPoolSize=workspace:8192  # 8GB instead of 6GB
```

### Low FPS / High Latency

1. Check power mode:
   ```bash
   nvpmodel -q  # Should show MAXN_SUPER
   ```

2. Check GPU clock:
   ```bash
   cat /sys/devices/platform/gpu.0/devfreq/*/cur_freq  # Should be 1020000000
   ```

3. Re-run jetson_clocks:
   ```bash
   sudo jetson_clocks
   ```

4. Check temperature (throttling?):
   ```bash
   cat /sys/class/thermal/thermal_zone*/temp
   # Values in millidegrees (e.g., 48000 = 48°C)
   ```

### Camera Busy Error

Kill existing demo processes:
```bash
pkill -f "demo_web"
```

---

## Files Reference

| File | Description |
|------|-------------|
| `export_onnx.py` | Export PyTorch model to ONNX |
| `demo_web_trt_pro.py` | 224px TensorRT demo with web UI |
| `demo_web_trt_320.py` | 320px TensorRT demo with web UI |
| `yowo_multitask_fp16.engine` | 224px TensorRT engine |
| `yowo_multitask_320_fp16_v2.engine` | 320px optimized TensorRT engine |

---

## Recommendations

For **smart home / action detection** use case:
- **320px at 8 FPS** is optimal - better detection quality, sufficient frame rate for slow actions
- Actions like "sitting", "walking", "cooking" don't need high FPS
- Detection quality matters more than speed for these applications

For **real-time applications** requiring smooth video:
- Use **224px at 14 FPS**
- Or upgrade to Orin NX/AGX for better performance
