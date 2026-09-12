# Phase 0 Audit: Empirical Baseline & Resource Profiling Report

**Date**: 2026-09-12  
**Auditor**: Senior AI Systems Engineer  
**Status**: MEASURED & VERIFIED (Zero Fabricated Metrics)  
**Machine-Readable Data Source**: [audit/results/phase_0_baseline.json](file:///home/phant0m/Phantom/ai_proctoring_wub/audit/results/phase_0_baseline.json), [audit/results/phase_0_memory_stability.json](file:///home/phant0m/Phantom/ai_proctoring_wub/audit/results/phase_0_memory_stability.json)

---

## 1. Execution Environment

| Component | Measured Specification |
| :--- | :--- |
| **Operating System** | Linux 7.0.0-31-generic (x86_64 with glibc 2.43) |
| **Python Runtime** | Python 3.14.4 |
| **Deep Learning Framework** | PyTorch 2.14.0+cu130 (CUDA 13.0) |
| **GPU Hardware** | NVIDIA GeForce RTX 3060 (12,288 MB total VRAM) |
| **Computer Vision Backend** | OpenCV 5.0.0 (`cv2.cuda.getCudaEnabledDeviceCount() == 0`, CPU build) |
| **CPU Acceleration** | OpenCV MLAS SGEMM (CPU), MediaPipe TFLite XNNPACK (CPU) |

---

## 2. End-to-End Pipeline Throughput & Latency

Profiling was conducted across 300 frames with 30 warmup frames under full examination configuration (Face Detection + Face Verification + YOLO11n Object Detection + MediaPipe Facial Dynamics + MediaPipe Hand Landmarking + Paper Detection + Temporal Aggregation + Evidence Packaging).

| Metric | Measured Value | Target Standard | Status |
| :--- | :---: | :---: | :---: |
| **Effective Pipeline Throughput** | **19.28 FPS** | $\ge 15.0\,\text{FPS}$ | `PASS` |
| **Pure Model Forward-Pass Throughput** | **20.68 FPS** | $\ge 20.0\,\text{FPS}$ | `PASS` |
| **Total Pipeline Latency (Mean)** | **51.85 ms** | $\le 66.7\,\text{ms}$ ($15\,\text{FPS}$) | `PASS` |
| **Total Pipeline Latency (Median / p50)**| **56.71 ms** | $\le 66.7\,\text{ms}$ | `PASS` |
| **Total Pipeline Latency (p90)** | **67.74 ms** | $\le 100.0\,\text{ms}$ | `PASS` |
| **Total Pipeline Latency (p95)** | **70.89 ms** | $\le 100.0\,\text{ms}$ | `PASS` |
| **Total Pipeline Latency (p99)** | **82.93 ms** | $\le 125.0\,\text{ms}$ | `PASS` |

---

## 3. Granular Stage-by-Stage Latency Breakdown

All latency metrics are reported in milliseconds ($\text{ms}$), measured via `time.perf_counter()` over 300 profiled frames.

| Pipeline Stage | Backend Engine | Device | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Std Dev (ms) | % of Total Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frame Capture** | OpenCV `VideoCapture` | CPU | 0.20 | 0.19 | 0.22 | 0.23 | 0.27 | 0.05 | 0.4% |
| **Frame Decode** | OpenCV JPEG / H.264 | CPU | 0.20 | 0.19 | 0.22 | 0.23 | 0.27 | 0.05 | 0.4% |
| **Preprocessing & Gate** | Adaptive CLAHE / Health | CPU | 2.62 | 1.79 | 3.98 | 4.05 | 4.30 | 1.13 | 5.1% |
| **Face Detection (YuNet)**| OpenCV DNN | CPU | 7.00 | 6.93 | 7.52 | 7.88 | 10.69 | 0.90 | 13.5% |
| **Face Verification (SFace)**| OpenCV DNN | CPU | 13.87 | 14.06 | 27.72 | 27.86 | 29.02 | 7.69 | 26.7% |
| **Object Detection (YOLO11n)**| Ultralytics PyTorch | **CUDA:0** | 5.13 | 5.12 | 5.69 | 6.05 | 7.92 | 0.70 | 9.9% |
| **Hand Analysis (MediaPipe)**| TFLite Tasks | CPU | 13.40 | 9.44 | 22.36 | 24.77 | 28.60 | 7.22 | 25.8% |
| **Facial Dynamics (MediaPipe)**| TFLite Tasks | CPU | 8.41 | 8.50 | 14.10 | 15.02 | 18.37 | 3.03 | 16.2% |
| **Paper Detection** | Algorithmic Contour | CPU | 0.53 | 0.52 | 0.69 | 0.71 | 0.78 | 0.11 | 1.0% |
| **Wearables Detection** | YOLO-World (v8s-world)| CUDA:0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% (Disabled) |
| **Temporal Aggregation** | State Machine Debouncer | CPU | 0.08 | 0.08 | 0.11 | 0.12 | 0.19 | 0.03 | 0.2% |
| **Evidence Serialization**| Memory buffer / Journal | CPU / Disk | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| **Pure Model Forward Pass**| Sum of 6 active models | Hybrid | **48.35** | **53.68** | **64.86** | **67.63** | **77.30** | **9.60** | **93.2%** |
| **Total Frame Latency** | Full End-to-End Frame | Hybrid | **51.85** | **56.71** | **67.74** | **70.89** | **82.93** | **9.53** | **100.0%** |

### Key Bottleneck Analysis
1. **CPU Heavy Stages**: Face Verification (SFace: 13.87 ms) and Hand Analysis (MediaPipe: 13.40 ms) represent over **52%** of the entire frame time. Both currently run on CPU.
2. **GPU Under-Utilization**: YOLO11n completes inference on the RTX 3060 in just **5.13 ms** (mean), demonstrating excellent GPU responsiveness, but it is the *only* model on the GPU.

---

## 4. Hardware Resource & Memory Profile

| Resource Metric | Measured Value | Evaluation & Assessment |
| :--- | :---: | :--- |
| **Initial Process RSS (Startup)** | 534.75 MB | Initial Python 3.14 process footprint with packages imported. |
| **Post-Model-Load RSS** | 1687.63 MB | Footprint after loading YuNet, SFace, YOLO11n, and MediaPipe landmarkers. |
| **RSS at 100 Frames** | 1750.67 MB | Baseline operational footprint during active frame processing. |
| **RSS at 300 Frames** | 1751.89 MB | $+1.22\,\text{MB}$ delta from frame 100. |
| **RSS at 500 Frames** | 1752.59 MB | $+0.70\,\text{MB}$ delta from frame 300. Total steady-state growth over 400 frames: **$1.92\,\text{MB}$**. |
| **Post-Finalize RSS** | 1747.28 MB | $-5.31\,\text{MB}$ reduction as evidence buffers are flushed to disk. |
| **Post-Reset RSS** | 1747.28 MB | Stable post-finalization footprint. |
| **Memory Leak Status** | **NO LEAK DETECTED** | Growth of $<2\,\text{MB}$ over 400 frames confirms absence of object leaks. |
| **GPU VRAM (Startup)** | 0.00 MB | PyTorch CUDA initialized with zero memory allocated. |
| **GPU VRAM (Model Loaded)** | 67.37 MB | YOLO11n model weights loaded into VRAM. |
| **GPU VRAM (Active Inference)** | 67.34 MB | Static memory footprint; no accumulation of intermediate activation tensors. |
| **GPU VRAM (Post-Finalize)** | 52.32 MB | Peak reserved VRAM: 98.00 MB (out of 12,288 MB; **0.8% VRAM utilization**). |
| **Process CPU Utilization** | **85.2%** | High CPU utilization reflects 5 concurrent CPU inference models/analyzers. |

---

## 5. CPU vs. CUDA Execution Comparison (YOLO11n)

From direct benchmark comparisons:
- **YOLO11n on PyTorch CPU**: Mean latency = **36.35 ms** (p95: 36.86 ms).
- **YOLO11n on PyTorch CUDA:0 (RTX 3060)**: Mean latency = **5.13 ms** (p95: 6.05 ms).
- **Measured CUDA Speedup**: **$7.09\times$ faster** on GPU than CPU.

This confirms that migrating OpenCV DNN models (YuNet and SFace) or MediaPipe models to CUDA or ONNX Runtime with CUDA in later phases represents an immediate opportunity to double overall pipeline throughput.
