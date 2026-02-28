# PointSS

**Point Cloud Semantic Segmentation with GGAM and ASD-SSM**

>  This is a preview release for paper review purposes. The complete model implementation will be released upon paper acceptance.

This repository contains the implementation of our point cloud semantic segmentation framework PointSS. Our approach achieves state-of-the-art performance on multiple 3D semantic segmentation benchmarks including  S3DIS.


## Installation

### Requirements
- Ubuntu 18.04 or above
- CUDA 11.3 or above
- PyTorch 1.10.0 or above
- Python 3.8+

### Environment Setup

**Step 1**: Create a conda environment
```bash
conda create -n pointss python=3.8 -y
conda activate pointss
```

**Step 2**: Install PyTorch (CUDA 11.8 example)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step 3**: Install dependencies
```bash
# Basic dependencies
pip install h5py pyyaml sharedarray tensorboard einops scipy plyfile

# PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-geometric

# Pointops (for efficient point operations)
cd libs/pointops
python setup.py install
cd ../..
```

**Step 4**: Install SpConv (for sparse convolution)
```bash
pip install spconv-cu118  # for CUDA 11.8
```

**Optional**: Install Flash Attention (CUDA >= 11.6 required)
```bash
pip install flash-attn --no-build-isolation
```

## Data Preparation

### ScanNet
Download ScanNet dataset and place it in `data/scannet/` directory:
```
data/
└── scannet/
    ├── scans/
    ├── scans_test/
    └── ...
```

Preprocess the data:
```bash
python tools/preprocess_scannet.py
```

### S3DIS
Download S3DIS dataset and place it in `data/s3dis/` directory:
```
data/
└── s3dis/
    ├── Stanford3dDataset_v1.2_Aligned_Version/
    └── ...
```

Preprocess the data:
```bash
python tools/preprocess_s3dis.py
```

### nuScenes
Download nuScenes dataset and place it in `data/nuscenes/` directory. Follow the official preprocessing steps.

## Training

### Train with single GPU
```bash
python tools/train.py --config-file configs/scannet/semseg-pt-v3m1-0-base.py
```

### Train with multiple GPUs
```bash
sh scripts/train.sh -p python -d scannet -c semseg-pt-v3m1-0-base -n exp_name -g 4
```

**Parameters:**
- `-p`: Python interpreter (default: python)
- `-d`: Dataset name (scannet/s3dis/nuscenes)
- `-c`: Config file name (without .py extension)
- `-n`: Experiment name
- `-g`: Number of GPUs

### Branch-specific Training

**GGAM + ASD-SSM** (main branch):
```bash
sh scripts/train.sh -d scannet -c your-config -n asd-ssm-exp -g 4
```


## Testing

### Test on validation set
```bash
python tools/test.py --config-file configs/scannet/semseg-pt-v3m1-0-base.py --options weight=path/to/checkpoint.pth
```

### Test with multiple GPUs
```bash
sh scripts/test.sh -d scannet -c semseg-pt-v3m1-0-base -n exp_name -g 4
```

## Model Architecture

Our framework is built upon **Point Transformer V3** with the following key innovations:

- **GGAM (Global-Guided Attention Module)**: Enhances feature learning with global context guidance
- **ASD-SSM (Adaptive State Space Module)**: Efficient long-range dependency modeling

Main model code: `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`

## Results

| Dataset | mIoU | Config |
|---------|------|--------|
| ScanNet | TBD | `configs/scannet/semseg-pt-v3m1-0-base.py` |
| S3DIS | TBD | `configs/s3dis/semseg-pt-v3m1-0-base.py` |
| nuScenes | TBD | `configs/nuscenes/semseg-pt-v3m1-0-base.py` |

## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{pointss2024,
  title={PointSS},
  author={XinWang,Xinyuan Zhang},
  year={2026}
}
```

## Acknowledgements

This codebase is built upon [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) and [Pointcept](https://github.com/Pointcept/Pointcept). We thank the authors for their excellent work.

## License

This project is released under the MIT License.
