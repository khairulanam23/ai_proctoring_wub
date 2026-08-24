# Notebooks & Colab Experiments

## Purpose
This directory is intended for exploratory data analysis, algorithm prototyping, visual evaluations, and Google Colab experiment notebooks.

## Guidelines & Best Practices

1. **Experimental Focus**:
   * Notebooks in this directory are reserved for research, hyperparameter exploration, and rapid prototyping.
   * All stable, tested, and reusable application logic, pipelines, and utility functions must be ported into the `src/` Python package.

2. **No Secrets or Credentials**:
   * Never hardcode API keys, passwords, authentication tokens, or sensitive credentials inside notebook cells or outputs.
   * Clear all outputs containing private environment paths or sensitive data before saving/committing.

3. **Data & Model Weight Hygiene**:
   * **Do not commit datasets**: Raw or processed training/validation datasets must remain in external storage or Google Drive.
   * **Do not commit model weights**: Large model checkpoints (`.pt`, `.pth`, `.onnx`, `.safetensors`, etc.) must be saved to Google Drive or model registries.

4. **Persistence via Google Drive**:
   * During Google Colab GPU sessions, persistent storage (datasets, model checkpoints, training logs, and benchmark reports) should be mounted and written to Google Drive under the designated `AI_Proctoring/` directory structure.
