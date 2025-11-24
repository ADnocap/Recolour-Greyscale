# DDColor: Image Colorization Implementation

A PyTorch implementation of **DDColor**, a state-of-the-art deep learning model for automatic image colorization using transformer-based dual decoders.

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2212.11613)
[![GitHub](https://img.shields.io/badge/Original-Repository-blue)](https://github.com/piddnad/DDColor)

---

## Overview

This project implements DDColor, a novel approach to image colorization that addresses the fundamental challenge of the ill-posed nature of the colorization problem. Given a grayscale image, DDColor predicts realistic and vibrant color information using a query-based transformer architecture with dual decoders.

### Why DDColor?

Traditional colorization methods often produce desaturated, unrealistic colors because they average multiple plausible color solutions. DDColor overcomes this by:

- Using **learnable color queries** that automatically discover diverse color patterns
- Processing features at **multiple scales** to reduce color bleeding
- Employing a **dual decoder architecture** for semantic-aware colorization
- Introducing a **colorfulness loss** to encourage vibrant outputs

---

## Key Features

### 1. Query-Based Color Learning

Instead of manually designing color priors, DDColor uses **100 learnable color queries** that automatically specialize in different color patterns through training.

### 2. Multi-Scale Feature Processing

By leveraging features at three different scales $(1/16, 1/8, 1/4)$, the model captures both fine details and global context, significantly reducing color bleeding across object boundaries.

### 3. Dual Decoders

- **Pixel Decoder**: Handles spatial structure and resolution restoration
- **Color Decoder**: Provides semantic-aware color understanding via cross-attention and self-attention mechanisms

### 4. Colorfulness Loss

A novel loss function that encourages more vibrant, visually appealing results by measuring color statistics in the generated images.

---

## Architecture

The DDColor architecture consists of four main components:

### 1. Encoder (Backbone)

- **ConvNeXt-L** extracts semantic features from grayscale images
- Produces multi-scale feature maps at $H/4×W/4, H/8×W/8, H/16×W/16,$ and $H/32×W/32$

### 2. Pixel Decoder

- Gradually upsamples features through 4 stages using PixelShuffle layers
- Creates a multi-scale feature pyramid with skip connections from the encoder

### 3. Color Decoder (Novel Component)

- Query-based transformer with 100 learnable color embeddings
- Consists of 3M color decoder blocks (where M=3)
- Each block performs:
  - **Cross-attention**: Color queries attend to image features
  - **Self-attention**: Color queries interact with each other
  - **MLP**: Feed-forward processing

### 4. Fusion Module

- Combines pixel decoder output (image embedding) and color decoder output (color embedding)
- Generates final AB color channels in LAB space

---

## Technical Details

### Color Space

DDColor operates in **LAB color space**:

- **L channel**: Luminance (the input grayscale image)
- **A channel**: Green ↔ Red color dimension
- **B channel**: Blue ↔ Yellow color dimension

### Input/Output

- **Input**: Grayscale image $x_L ∈ R^{H×W×1}$
- **Output**: Color channels $ŷ_{AB} ∈ R^{H×W×2}$
- **Final Result**: Concatenate $[x_L, ŷ_{AB}]$ → LAB image $(H×W×3)$ → Convert to RGB

### Loss Functions

1. **Smooth L1 Loss**: For pixel-level color prediction
2. **Perceptual Loss**: Using VGG features for semantic similarity
3. **Adversarial Loss**: For realistic color generation
4. **Colorfulness Loss**: Encourages vibrant, saturated colors

---

## Understanding the Problem

### The Ill-Posed Nature of Colorization

Colorization is fundamentally **ill-posed** because:

- Multiple color images can produce the same grayscale image
- No unique solution exists without additional context
- Example: A grayscale value of 128 could represent dark red, dark green, dark blue, or any combination

### Multi-Modal Uncertainty

Objects in images can have multiple plausible colors:

- A car could be red, blue, black, white, or silver
- A shirt could be any fabric color
- A wall could be any paint color

**The Challenge**: Naive neural networks tend to **average** all possible solutions, resulting in desaturated, unrealistic colors.

**DDColor's Solution**: Uses semantic understanding via the color decoder and multi-scale features to make contextually appropriate, decisive color choices.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Required Packages

```bash
# Core dependencies
pip install torch torchvision torchaudio

# Image processing and utilities
pip install opencv-python pillow numpy pyyaml scipy scikit-image

# Deep learning utilities
pip install timm tensorboard

# Additional dependencies
pip install lmdb lpips
```

---


## Usage

### Training

1. **Prepare your dataset**: Organize RGB images for training

2. **Configure training parameters** in the notebook or configuration file:

   - Input size: 256×256 (default)
   - Batch size: 4-8 (depending on GPU memory)
   - Learning rate: 1e-4
   - Number of epochs/iterations

3. **Run training**:

   ```python
   # Execute the training loop
   train()
   ```

4. **Monitor progress**:
   - Checkpoints saved periodically
   - Sample colorizations generated during training
   - TensorBoard logs for loss visualization

### Inference

```python
# Colorize a single image
result = inference(
    image_path='path/to/grayscale/image.jpg',
    checkpoint_path='checkpoints/latest.pth'
)
```

---

## Training Details

### Optimization

- **Generator**: AdamW optimizer with cosine annealing schedule
- **Discriminator**: Adam optimizer with linear decay
- **Warm-up**: Initial learning rate warm-up for stable training

### Data Augmentation

- Random horizontal flipping
- Random cropping
- Color jittering (applied to RGB before conversion to grayscale)

### Multi-GPU Training

- Supports DataParallel for multi-GPU setups
- Automatic gradient accumulation for large batch sizes

---

## Results

During training, the model generates periodic visualizations showing:

- Input grayscale image
- Predicted colorization
- Ground truth color image

Checkpoints are saved at regular intervals, allowing you to:

- Resume training from any point
- Evaluate different training stages
- Select the best performing model

---

## Key Concepts Explained

### Why LAB Color Space?

LAB separates luminance (L) from color information (A, B), making it ideal for colorization:

- The L channel directly corresponds to the grayscale input
- Only the A and B channels need to be predicted
- Perceptually uniform color representation

### Colorfulness Loss

Measures the vibrancy of generated colors:

```
Colorfulness = σ(AB) + 0.3 × |μ(AB)|
```

Where σ is standard deviation and μ is mean of the AB channels. This encourages the model to generate saturated, vivid colors rather than gray or muted tones.

---

## Citation

If you use this code or find it helpful, please cite the original DDColor paper:

```bibtex
@article{kang2022ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
  journal={arXiv preprint arXiv:2212.11613},
  year={2022}
}
```

---

## References

- **Paper**: [DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders](https://arxiv.org/pdf/2212.11613)
- **Original Implementation**: [GitHub Repository](https://github.com/piddnad/DDColor)

---

## License

This implementation is for educational and research purposes. Please refer to the original DDColor repository for licensing information.

---

## Acknowledgments

This implementation is based on the DDColor paper by Kang et al. Special thanks to the authors for their innovative work in the field of image colorization.
