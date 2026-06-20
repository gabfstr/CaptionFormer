# Installation

CaptionFormer is built on [Detectron2](https://github.com/facebookresearch/detectron2) and [Mask2Former](https://github.com/facebookresearch/Mask2Former).

## Requirements
- Linux, NVIDIA GPU, conda, gcc ≤ 12
- Python 3.8, PyTorch 2.1.0, CUDA 12.1 for bit-identical, or other combination.
- Detectron2 : follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html)
- `pip install -r requirements.txt`

## Example conda environment setup
```bash
# 1. Conda env + CUDA + PyTorch
conda create -n captionformer python=3.8 -y
conda activate captionformer
conda install -c nvidia/label/cuda-12.1.0 cuda -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

# 2. Install Detectron2 under your working directory
git clone --branch v0.6 --depth 1 https://github.com/facebookresearch/detectron2.git
pip install ninja
pip install --no-build-isolation -e ./detectron2/

# 3. Python dependencies
git clone https://github.com/<USER>/CaptionFormer.git
cd CaptionFormer
pip install -r requirements.txt

# 4. Compile the MS-DeformAttn CUDA op (on GPU)
cd captionformer/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```

## Verify install
```python
python -c "import torch, detectron2, captionformer, captionformer_video, MultiScaleDeformableAttention; print('OK')"
```

## Next

Dataset preparation and processing - see [`DATA_PREPARATION.md`](DATA_PREPARATION.md).