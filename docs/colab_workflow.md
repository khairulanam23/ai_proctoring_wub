# Google Colab & Google Drive Experiment Workflow

## Overview
This document defines the hybrid development and experimentation workflow connecting local development (Antigravity IDE), version control (GitHub), GPU-accelerated computing (Google Colab), and persistent artifact storage (Google Drive).

---

## Architecture & Data Flow

```text
Local Antigravity IDE (CPU) ──[ git push ]──> GitHub Repository
                                                    │
                                             [ git clone / pull ]
                                                    ↓
                                            Google Colab (GPU)
                                                    │
                                           [ mount / read / write ]
                                                    ↓
                                            Google Drive Storage
                                       (Datasets, Models, & Results)
```

---

## Google Drive Structure

Persistent data and large artifacts reside in Google Drive under a dedicated root directory:

```text
AI_Proctoring/
├── datasets/          # Raw and preprocessed training/validation datasets
│   ├── face_detection/
│   ├── head_pose/
│   └── audio_events/
├── models/            # Trained weights, checkpoints, ONNX exports (.pt, .onnx, .safetensors)
│   ├── checkpoints/
│   └── final/
├── experiments/       # TensorBoard logs, training metrics, run configurations
└── results/           # Evaluation reports, confusion matrices, benchmark plots
```

---

## Colab Session Lifecycle

When initiating a GPU experimentation session on Google Colab, follow this standardized sequence:

### 1. Mount Google Drive
Mount persistent storage to make datasets and model checkpoints accessible:

```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = '/content/drive/MyDrive/AI_Proctoring'
```

### 2. Clone the GitHub Repository
Clone or pull the latest source code from the project repository:

```bash
!git clone https://github.com/khairulanam23/ai_proctoring_wub.git
%cd ai_proctoring_wub
```

### 3. Install Dependencies
Install the required development dependencies inside the Colab runtime:

```bash
!pip install -r requirements.txt
```

### 4. Execute Experimentation
* Read datasets directly from `/content/drive/MyDrive/AI_Proctoring/datasets/`.
* Write model checkpoints and training logs to `/content/drive/MyDrive/AI_Proctoring/models/` and `experiments/`.
* Save final evaluation metrics and plots to `/content/drive/MyDrive/AI_Proctoring/results/`.

### 5. Sync Code Refinements
* Any algorithmic improvements or reusable utility functions developed during Colab experimentation should be refactored into the `src/` directory and committed back to GitHub.

---

## Strict Separation of Concerns

| Asset Type | Storage Target | Tracked in Git? |
| :--- | :--- | :--- |
| Python Source Code (`src/`) | GitHub | **Yes** |
| Configuration Files (`configs/`) | GitHub | **Yes** |
| Unit & Integration Tests (`tests/`) | GitHub | **Yes** |
| Documentation (`docs/`) | GitHub | **Yes** |
| Notebook Templates (`notebooks/`) | GitHub | **Yes** |
| Raw / Processed Datasets | Google Drive (`datasets/`) | **NO** (Gitignored) |
| Model Weights (`.pt`, `.onnx`, etc.) | Google Drive (`models/`) | **NO** (Gitignored) |
| Experiment Logs & Checkpoints | Google Drive (`experiments/`) | **NO** (Gitignored) |
| Large Benchmark Outputs | Google Drive (`results/`) | **NO** (Gitignored) |

> [!WARNING]
> Never commit large binary datasets, model weights, or sensitive tokens to GitHub. Keep all heavy artifacts synchronized through Google Drive.
