# Progressive Prompt Detailing for Improved Alignment in Text-to-Image Generative Models (SCoPE)

Ketan Suhaas Saichandran*, Xavier Thomas*, Prakhar Kaushik, Deepti Ghadiyaram  
(*Equal Contribution)

[📄 Paper (arXiv)](https://arxiv.org/abs/2503.17794)

---

This repository contains the code for **SCoPE**, a training-free method to improve prompt-image alignment in text-to-image diffusion models by progressively interpolating between coarse-to-fine prompt embeddings during the denoising process.

SCoPE can be easily applied on top of existing diffusion pipelines without any retraining.

View `run.sh` for an example to run our pipeline

---


### 🔧 Setting up `t2v_metrics` for VQA Scoring

To use the VQA scoring model (`clip-flant5-xxl`), follow these steps to install `t2v_metrics` in a dedicated conda environment.

#### 📁 Step 1: Clone the repository

```bash
cd scorers/
git clone https://github.com/linzhiqiu/t2v_metrics
cd t2v_metrics
```

#### 🐍 Step 2: Create and activate a new conda environment

```bash
conda create -n t2v python=3.10 -y
conda activate t2v
```

#### 📦 Step 3: Install dependencies

```bash
conda install pip -y
pip install torch torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git
```

#### 🛠️ Step 4: Install `t2v_metrics` locally

```bash
pip install -e .
```

You’re now ready to use the VQA scoring functionality in `get_scores.py`.