# FI-Net: Towards Robust Identification in Dense  Scenes via Re-ID-Augmented Face Feature Fusion

A deep learning system for multi-image face recognition via feature fusion, integrating state-of-the-art face recognition and fusion techniques.

## 🚀 Features

- **Multi-image feature fusion**: Supports methods like ProxyFusion and CAFace
- **Face quality assessment**: Integrated RG-FIQA quality scoring network
- **Dynamic feature pooling**: Pose-aware dynamic pooling for robust aggregation

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (recommended)
- 8GB+ GPU memory (recommended)

## 🛠️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/FI-Net.git
cd FI-Net
```

### 2. Create a virtual environment
```bash
conda create -n fi-net python=3.8
conda activate fi-net
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install the project (editable)
```bash
pip install -e .
```

## 📁 Project Structure

```
FI-Net/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── LICENSE                      # Open-source license
├── .gitignore                   # Git ignore rules
├── face_fusion/                 # Face feature fusion methods
│   ├── caface-master/          # CAFace implementation
│   └── ProxyFusion-NeurIPS-main/ # ProxyFusion implementation
├── rg_fiqa/                     # Face quality assessment (RG-FIQA)
├── feature_pooling/             # Dynamic feature pooling
├── recognize/                   # Face recognition pipelines
├── reid/                        # Person re-identification
├── visualization/               # Visualization tools
```

## 🚀 Quick Start

### 1. Basic usage

```python
from fi_net import FaceRecognitionPipeline

# Initialize pipeline
pipeline = FaceRecognitionPipeline(
    model_type='proxyfusion',
    quality_assessment=True
)

# Load a model
pipeline.load_model('path/to/checkpoint.pth')

# Run recognition
results = pipeline.recognize(face_images)
```

### 2. Train

```bash
# Train with default config
python scripts/train.sh

# Train with a custom config
python fi_net/training/train.py --config configs/custom.yaml
```

### 3. Evaluate

```bash
# Run all evaluations
python run_all_evals.py

# Evaluate a specific component
python fi_net/evaluation/eval_proxyfusion.py
```

## 📊 Results

### Metrics

| Method | TAR@FAR=1e-3 | TAR@FAR=1e-4 | Latency (ms) |
|------|-------------|-------------|-------------|
| Baseline | 85.2% | 78.1% | 12.3 |
| ProxyFusion | 89.7% | 83.4% | 15.6 |
| ProxyFusion+Quality | 91.2% | 85.8% | 18.2 |
| Dynamic Pooling | 88.9% | 82.1% | 14.1 |

### Visualizations

This repo includes multiple visualization utilities, such as:
- t-SNE feature distribution plots
- ROC curve comparisons
- Network structure diagrams


## 📖 Citation
If you find this work useful, please cite our paper:
```bash
@article{zhang2025finet,
  author = {Zhang, Qichao and Chen, Genlang and Zhang, Jiajian and Jia, Chengcheng},
  title = {{FI-Net: Towards Robust Identification in Dense Scenes via Re-ID-Augmented Face Feature Fusion}},
  url = {https://github.com/zjucgl/FI-Net},
  year = {2025}
}
```

## 🙏 Acknowledgements

Built on top of the following open-source projects:
- [CAFace](https://github.com/mk-minchul/CAFace) - Face feature fusion
- [ProxyFusion](https://github.com/your-proxyfusion-repo) - Proxy-based fusion
- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition framework

