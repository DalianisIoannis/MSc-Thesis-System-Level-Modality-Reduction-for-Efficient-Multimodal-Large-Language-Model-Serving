# MSc-Thesis: System-Level Modality Reduction for Efficient Multimodal Large Language Model Serving

## 📌 Project Overview

This repository contains the thesis implementation and benchmark infrastructure for evaluating **system-level modality reduction techniques** to optimize the serving of Multimodal Large Language Models (MLLMs). The core insight is that intelligently reducing input modalities (images and videos) can dramatically improve serving efficiency without significantly compromising task quality.

### 🎯 Thesis Contribution

**Problem**: Deploying MLLMs in production faces critical bottlenecks:
- Transformer's quadratic complexity during the prefill phase
- Massive Key-Value (KV) cache memory consumption
- High latency and limited throughput under load

**Solution**: Implement system-level modality reduction techniques:

| Modality | Technique | Results |
|----------|-----------|---------|
| **Images** | Proportional resizing | ↓ 51% TTFT, ↓ 53% KV cache, ↓ ~4% accuracy |
| **Videos** | Scene-change intelligent frame sampling | ↓ 79% TTFT, ↓ 76% KV cache, maintained accuracy |

These optimizations can be applied at the system level without model retraining, making them immediately deployable in production MLLM serving infrastructure.

## 📊 LLMPerf Inference Benchmark Suite

LLMPerf is a comprehensive benchmark suite for measuring MLLM serving performance across diverse workloads and deployment scenarios. It provides rigorous evaluation infrastructure for understanding performance trade-offs.


### 🏗️ Key Implementation Details

#### 1. **Video Frame Sampling (`vllm/vllm/assets/video.py`)**

This is the **most critical contribution** for video optimization. The module provides multiple frame sampling strategies:

- **Scene Change Detection (`sample_frames_by_scene_change_open_cv_version`)**:
  - Uses `scenedetect.ContentDetector` to identify meaningful scene boundaries
  - Intelligently selects key frames: first, middle, and last of each scene
  - Dramatically reduces redundant frame processing compared to uniform sampling
  - Falls back to uniform sampling when no scenes are detected
  
- **Uniform Baseline Sampling (`_read_and_uniformly_sample_frames`)**:
  - Traditional approach for comparison
  - Evenly selects frames across the entire video duration

- **Extended Version (`sample_frames_by_scene_change_decord_version`)**:
  - Includes timing information suitable for external evaluation frameworks
  - Returns timestamps and video duration along with frames

#### 2. **Image Resizing & Modality Size Reduction (`llmperf/preprocessing/modality_size.py`)**

Implements proportional image downsizing as a system-level optimization:
- Adjustable resize ratios to find optimal speed/accuracy trade-offs
- Direct integration into the serving pipeline
- Minimal quality impact on most vision tasks

#### 3. **Benchmarking Infrastructure (`llmperf/`)**

Comprehensive benchmark suite with:
- **Config system**: Models, workloads, datasets, and approaches configurations
- **Preprocessing**: Modality path resolution, size reduction, input handling
- **Postprocessing**: Performance metric collection and output formatting
- **Workload generation**: Static, Poisson, Gamma, and RPS (rocks/pebbles/sand) distributions

### 📈 Experimental Pipeline

The benchmark follows this workflow:

```
1. Create workloads (scripts/create_*_workloads.py)
   ↓
2. Select optimization approach (uniform/scene-change sampling, resizing ratios)
   ↓
3. Run experiments with vLLM backend (scripts/run_*_workloads.py)
   ↓
4. Collect metrics: TTFT, KV cache usage, throughput, accuracy
   ↓
5. Analyze results and compare strategies
```

### 🔧 Key Experimental Scripts

- **`scripts/run_static_workloads_img_vid_resizing.py`**: Main experiment runner for image/video optimization
  - Supports multiple models (Qwen2-VL, LLaVA, Mistral, etc.)
  - Tests different resizing/sampling strategies
  - Measures performance across diverse tasks (detection, captioning, VQA, etc.)

- **`scripts/run_vllm_workloads.py`**: Baseline vLLM serving experiments

- **`scripts/create_static_workloads.py`**: Generates workload datasets for reproducible benchmarking
```
curl -fsSL https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init - bash)"' >> ~/.profile

echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

exec "$SHELL"
```
1. Clone repositories
    ```
    git clone git@gitlab.software.imdea.org:muse-lab/mllm-inference-workload-eval.git
    cd mllm-inference-workload-eval/
    git clone git@gitlab.software.imdea.org:muse-lab/vllm.git
    cd vllm/
    git checkout dev
    ```
2. Create virtual environemnt
    ```
    pyenv install 3.12.8
    pyenv virtualenv 3.12.8 vllm-v0.7.2
    pyenv activate vllm-v0.7.2
    ```
3. Install vLLM with multimodal support (for image, video, audio)
    ```
    cd vllm/
    # For precompiled wheels (faster)
    VLLM_USE_PRECOMPILED=1 pip install --editable .
    
    # Or for development version
    export VLLM_COMMIT=dc1b4a6f1300003ae27f033afbdff5e2683721ce
    export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
    pip install --editable .
    ```
    
    ⚠️ **Important**: This version includes modifications in `vllm/assets/video.py` for intelligent frame sampling.

4. Install LLMPerf benchmark suite
    ```
    cd ..
    pip install --editable .
    ```

5. (Optional) For vLLM development
    ```
    cd vllm
    pip install -r requirements/dev.txt
    pre-commit install --hook-type pre-commit --hook-type commit-msg
    ```

### Additional Dependencies

Key packages used in the video sampling implementation:
```bash
pip install scenedetect opencv-python av decord torch torchvision
```


## 📁 Project Structure & Key Files

### Core Optimization Implementation

- **[vllm/vllm/assets/video.py](vllm/vllm/assets/video.py)** ⭐
  - Scene change detection for intelligent video frame sampling
  - Multiple sampling strategies (scene-based, uniform, with/without timing info)
  - Key function: `sample_frames_by_scene_change_open_cv_version()`

- **[llmperf/preprocessing/modality_size.py](llmperf/preprocessing/modality_size.py)**
  - Image resizing and modality downsizing logic
  - Adjustable parameters for quality/performance trade-offs

### Benchmarking Infrastructure

- **[llmperf/config/](llmperf/config/)** - Configuration management
  - `models.py` - Supported MLLM models
  - `workloads.py` - Workload definitions
  - `datasets.py` - Dataset configurations
  - `approaches.py` - Serving approaches (isolation, vanilla vLLM)

- **[llmperf/preprocessing/](llmperf/preprocessing/)** - Input handling
  - Data loading, path resolution, modality preprocessing

- **[llmperf/postprocessing/](llmperf/postprocessing/)** - Metrics collection
  - Performance statistics (TTFT, throughput, KV cache, accuracy)

### Experiment Scripts

- **[scripts/run_static_workloads_img_vid_resizing.py](scripts/run_static_workloads_img_vid_resizing.py)** ⭐
  - Main experiment runner with image/video optimizations
  - Tests various resize ratios and sampling strategies
  - Integrates with vLLM for serving

- **[scripts/create_static_workloads.py](scripts/create_static_workloads.py)**
  - Workload dataset generation from raw data

### Analysis & Visualization

- **[exploration/eval_plots.py](exploration/eval_plots.py)**
  - Results visualization and statistical analysis
  - Generates comparative plots of optimization strategies

- **[exploration/eval-images.ipynb](exploration/eval-images.ipynb)** & **[exploration/eval-videos.ipynb](exploration/eval-videos.ipynb)**
  - Interactive analysis notebooks for image and video results

## ⚙️ Installation & Setup

⚠️ **Note**: This codebase is a research artifact demonstrating the thesis methodology. While individual components are functional, the complete end-to-end pipeline may require dataset preparation and environment configuration specific to your setup.

### Prerequisites

To install pyenv:



### 📦 Datasets & Workloads

Download datasets separately or use minimal versions provided. The benchmarks evaluate on:
- **Image tasks**: Object detection, captioning, visual QA (LLaVA-Bench, MMBench)
- **Video tasks**: Video QA, scene understanding, temporal reasoning (VideoMME, LLaVA-Video-Bench)

#### Workload Types

The benchmark supports multiple request arrival distributions:

```
# Static workloads (all requests at once, no arrival distribution)
python scripts/create_static_workloads.py

# Poisson distribution (realistic request arrivals)
python scripts/create_poisson_workloads.py

# Gamma distribution (bursty traffic patterns)
python scripts/create_gamma_workloads.py

# RPS classification (rocks/pebbles/sand for varying patterns)
python scripts/create_rps_workloads.py

# Multi-modality mixes
python scripts/create_multi_stream_workloads.py --workloads text-static image-static video-static --request-rates 0.05 0.1 0.25 ... 10.0
```

### 🚀 Running Experiments

#### Baseline: Vanilla vLLM (without optimizations)
```
python scripts/run_vllm_workloads.py
```

#### With Image/Video Modality Reduction
```
python scripts/run_static_workloads_img_vid_resizing.py
```

This script tests various optimization parameters:
- Image resize ratios (control quality/speed trade-off)
- Video frame sampling strategies (uniform vs. scene-change detection)
- Different MLLM models and task types

## 🎓 Thesis Results & Key Findings

### Performance Impact Summary

Evaluated on NVIDIA A100 GPUs with models like Qwen2-VL, LLaVA, and Mistral:

#### **Image Optimization (Proportional Resizing)**
- **Speed improvement**: Up to 51% reduction in Time-to-First-Token (TTFT)
- **Memory improvement**: Up to 53% reduction in KV cache footprint
- **Quality trade-off**: Minimal accuracy loss (~4%) across vision tasks
- **Sweet spot**: 50-70% of original resolution maintains quality while achieving significant speedup

#### **Video Optimization (Scene Change Detection)**
- **Speed improvement**: Up to 79% reduction in TTFT
- **Memory improvement**: Up to 76% reduction in KV cache usage
- **Quality trade-off**: Maintained competitive accuracy (often equal or better than uniform sampling)
- **Key advantage**: Scene-aware sampling captures semantically important frames, avoiding redundancy

### Evaluation Benchmarks

Results validated across diverse vision-language tasks:
- **Detection**: Object detection in images/videos
- **Captioning**: Image and video captioning
- **VQA**: Visual question answering
- **MC**: Multiple-choice video understanding
- **Spatial reasoning**: Scene understanding and temporal reasoning

### Why This Matters

✅ **System-level approach**: No model retraining required
✅ **Immediate deployment**: Drop-in optimization for existing MLLM serving infrastructure  
✅ **Scalability**: Multiplicative benefits when combined with other optimizations
✅ **Production-ready**: Tested on high-throughput vLLM engine with real workloads

## 🔍 What to Focus On

If you're exploring this codebase, start here:

1. **For understanding video sampling**: [vllm/vllm/assets/video.py](vllm/vllm/assets/video.py) - especially `sample_frames_by_scene_change_open_cv_version()`
2. **For understanding the benchmark**: [scripts/run_static_workloads_img_vid_resizing.py](scripts/run_static_workloads_img_vid_resizing.py)
3. **For image optimization**: [llmperf/preprocessing/modality_size.py](llmperf/preprocessing/modality_size.py)
4. **For results analysis**: [exploration/eval-images.ipynb](exploration/eval-images.ipynb) and [exploration/eval-videos.ipynb](exploration/eval-videos.ipynb)

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{dalianis2024modality,
  title={System-Level Modality Reduction for Efficient Multimodal Large Language Model Serving},
  author={Dalianis, Giannis},
  school={IMDEA Software Institute},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.