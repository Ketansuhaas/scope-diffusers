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