# Enterprise Multi-Stream Signal & Audio Spectral Processing Pipeline (cuFFT)

[![CUDA](https://img.shields.io/badge/CUDA-11.x%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![cuFFT](https://img.shields.io/badge/NVIDIA-cuFFT-76b900.svg)](https://developer.nvidia.com/cufft)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Jatin288-black)](https://github.com/Jatin288/cuda-cufft-spectral-pipeline)
[![Interactive Demo](https://img.shields.io/badge/Interactive-Visualizer-brightgreen)](https://jatin288.github.io/cuda-cufft-spectral-pipeline/)

A high-performance, enterprise-grade GPU signal processing pipeline developed as the **GPU Specialization Capstone Project**.

---

## 🔬 Project Overview

Real-world signal processing (such as speech processing, radar/sonar analysis, vibration monitoring, and biomedical EEG/ECG) requires computing spectral representations across hundreds of continuous high-rate channels in real time.

This project implements an end-to-end **Multi-Stream Short-Time Fourier Transform (STFT) and Spectral Noise Reduction Engine** in CUDA C++ using NVIDIA's **cuFFT** library and asynchronous **CUDA Streams**.

### Key Computational Stages:
1. **Asynchronous Memory Transfer:** Continuous input signal buffers are streamed to GPU memory via pinned host buffers (`cudaMallocHost` and `cudaMemcpyAsync`).
2. **Custom Windowing Kernel (`ApplyHammingWindowKernel`):** Applies a parallel periodic Hamming window $w(n) = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right)$ across STFT frames to minimize spectral leakage across discrete Fourier bins.
3. **Batched cuFFT Execution (`cufftExecR2C`):** Computes batched 1D Real-to-Complex Fast Fourier Transforms in the frequency domain.
4. **Custom Spectral Filtering & Magnitude Kernel (`SpectralMagnitudeAndFilterKernel`):**
   - Calculates the Power Spectral Density $|X(f)|^2$.
   - Applies an adaptive noise-gate attenuation threshold (attenuating noise components below $-35\text{ dB}$).
   - Extracts Decibel (dB) energy levels for 2D waterfall spectrogram reconstruction.
5. **Multi-Stream Overlap:** Dispatches concurrent signal streams across 4 independent `cudaStream_t` queues, saturating GPU Streaming Multiprocessors (SMs) and achieving full latency hiding between compute and PCIe data transfers.

---

## 📁 Repository Layout

```
.
├── src/
│   └── spectral_pipeline.cu   # Complete CUDA C++ pipeline with cuFFT & custom kernels
├── scripts/
│   ├── gen_signals.py         # Multi-harmonic audio & sensor signal generator
│   └── generate_artifacts.py  # Automated verification & telemetry generator
├── data/
│   ├── input/                 # Raw time-domain signal CSV files
│   └── output/                # Processed 2D spectrogram CSV matrices
├── docs/
│   ├── index.html             # Interactive HTML5 spectrogram visualizer & presentation
│   └── pipeline_trace.json    # Exported telemetry for web inspection
├── execution_log.txt          # Complete execution trace across 100 signal datasets
├── Makefile                   # Production build system targeting compute 6.0 - 8.6
├── run.sh                     # One-shot build, dataset generation, and execution script
├── LICENSE                    # MIT License
└── README.md                  # Comprehensive Capstone documentation
```

---

## ⚡ Multi-Stream Architecture

```
Host (CPU)                   CUDA Stream 0            CUDA Stream 1            CUDA Stream 2            CUDA Stream 3
┌────────────────┐          ┌───────────────┐        ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
│ Pinned Memory  │──Async──>│ H2D Memcpy    │        │               │        │               │        │               │
│ Batch Signals  │          │ Hamming Window│        │ H2D Memcpy    │        │               │        │               │
│                │          │ cuFFT ExecR2C │        │ Hamming Window│        │ H2D Memcpy    │        │               │
│                │          │ Spectral Gate │        │ cuFFT ExecR2C │        │ Hamming Window│        │ H2D Memcpy    │
│                │<──Async──│ D2H Memcpy    │        │ Spectral Gate │        │ cuFFT ExecR2C │        │ Hamming Window│
└────────────────┘          └───────────────┘        │ D2H Memcpy    │        │ Spectral Gate │        │ cuFFT ExecR2C │
                                                     └───────────────┘        │ D2H Memcpy    │        │ Spectral Gate │
                                                                              └───────────────┘        │ D2H Memcpy    │
                                                                                                       └───────────────┘
```

---

## 🚀 Building & Running

### Requirements
- NVIDIA GPU (Compute Capability $\ge 6.0$)
- NVIDIA CUDA Toolkit 11.x or higher with `libcufft`
- GCC/G++ 7+ (C++14 standard)
- Python 3.8+ (for dataset synthesis)

### One-Command Execution:
```bash
chmod +x run.sh
./run.sh
```

### Manual Compilation:
```bash
make -j$(nproc)
./bin/spectral_pipeline --input data/input --output data/output --streams 4 --threshold -35.0
```

#### Command-Line Options:
| Flag | Description | Default |
|---|---|---|
| `--input <dir>` | Directory containing signal CSV datasets | *(Required)* |
| `--output <dir>` | Directory to write output spectrogram CSV matrices | *(Required)* |
| `--streams <N>` | Number of concurrent asynchronous CUDA streams | `4` |
| `--fft-size <N>` | FFT window size in samples | `1024` |
| `--hop-size <N>` | STFT frame hop size (overlap) | `512` |
| `--threshold <dB>` | Noise-gate spectral cutoff threshold | `-35.0` |
| `--max <N>` | Maximum number of signals to process | `All` |

---

## 📊 Enterprise Performance Results (Tesla T4)

| Metric | Single-Stream Serialized | 4-Stream Concurrency (Ours) | Speedup |
|---|---|---|---|
| **Average Latency / Signal** | 5.82 ms | **1.68 ms** | **3.46x** |
| **Spectral Throughput** | 2.81 MS/s | **9.75 Megasamples/sec** | **3.47x** |
| **PCIe Transfer Overlap** | 0% (Blocking) | **~85% compute/transfer overlap** | — |
| **Memory Bandwidth Efficiency** | Standard Pageable | **Pinned Memory (cudaMallocHost)** | **~2.1x bandwidth** |

---

## 🌐 Interactive Presentation & Demonstration

An interactive, web-based demonstration and presentation dashboard is included:

👉 **[Launch Interactive Capstone Visualizer](https://jatin288.github.io/cuda-cufft-spectral-pipeline/)**

Features:
- Time-domain waveform canvas and audio playback preview
- Interactive 2D Waterfall Spectrogram with Magma colormap
- Live CUDA stream timeline diagram demonstrating asynchronous concurrency
- Presentation walkthrough slides summarizing goals, challenges, and conclusions

---

## 📜 License
MIT © 2024 [Jatin Pawar](https://github.com/Jatin288)
