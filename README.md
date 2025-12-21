# Automatic Image Colorization using CNNs

A systematic exploration of CNN-based approaches for automatic grayscale image colorization, comparing different architectures, color spaces, and loss formulations.

**Authors:** Alexandre Dalban & Wacil Lakbir  
**Institution:** CentraleSupélec

## Overview

Image colorization—predicting plausible colors from grayscale images—is challenging for machine learning systems due to its ill-posed nature: a single grayscale intensity can correspond to many different valid colors. This project conducts a systematic investigation of CNN-based colorization approaches across three key dimensions:

1. **Architecture**: Simple encoder-decoder CNN versus U-Net architectures with skip connections
2. **Color Space**: RGB versus LAB (separating luminance from chrominance)
3. **Problem Formulation**: Regression with continuous outputs versus classification over discretized color bins

## The Color Averaging Problem

A fundamental challenge arises when using standard regression losses like Mean Squared Error (MSE). Consider a grayscale pixel with intensity 128 that corresponds to three different colors in training: red (255,0,0), green (0,255,0), and blue (0,0,255). MSE loss optimizes for the conditional mean:

```
f*(128) = E[RGB | I_gray = 128] = (85, 85, 85)
```

This produces a muddy brown color that averages the three pure hues, rather than committing to one plausible option. This "color averaging" problem produces the characteristic desaturated, brownish outputs of naive regression approaches.

## Architectures

### Baseline CNN Encoder-Decoder

A simple encoder-decoder design with 4.3M parameters:

- **Encoder**: 4 downsampling blocks (Conv + BatchNorm + ReLU + MaxPool), 96×96 → 6×6, channels 1 → 512
- **Decoder**: 4 upsampling blocks (ConvTranspose + BatchNorm + ReLU), 6×6 → 96×96, channels 512 → 3

**Limitation**: Strong bottleneck compression leads to loss of fine spatial details, producing blurry and desaturated colors.

### U-Net Architecture

U-Net adds skip connections between encoder and decoder layers at matching resolutions via channel-wise concatenation. This preserves fine-grained spatial information lost during downsampling, enabling sharper colorization boundaries.

**Two variants:**

- **U-Net RGB**: 4.5M parameters, 96×96 resolution, 4-level encoder-decoder
- **U-Net LAB**: 20.5M parameters, 512×512 resolution, deeper architecture

## Color Space Representations

### RGB Approach

Model receives single-channel grayscale and predicts all three RGB channels. The network must jointly learn brightness and color information.

### LAB Approach

LAB color space separates luminance (L channel, 0-100) from chrominance (A and B channels, -128 to +127). We extract the L channel as input and train the model to predict only AB channels.

**Pipeline:** RGB → LAB → extract L [0,1] as input → predict AB [-1,1] → recombine [L, predicted AB] → RGB

This decoupling simplifies learning by removing brightness prediction, allowing the model to focus capacity on color.

## Loss Functions

### MSE vs L1

- **MSE Loss**: Optimizes for conditional mean → color averaging
- **L1 Loss**: Optimizes for conditional median → sharper colors but doesn't solve multi-modality

### Combined Loss

Weighted combination balances smoothness (MSE) with sharpness (L1):

```
L_total = α × MSE + (1-α) × L1,  where α ∈ {0.3, 0.5, 0.7}
```

Balanced weighting (α = 0.5-0.7) provides good trade-offs between saturation and stability.

## Classification-Based Approach

To better handle multi-modal color distributions, we reformulate colorization as classification over discretized color bins.

**Method:**

- Partition AB space [-128, 127] into grid with bin size 10 → 26×26 = 676 bins
- Same U-Net encoder-decoder, output modified to predict probability distribution over 676 bins (softmax)
- Weighted cross-entropy loss to handle class imbalance
- Inference: argmax over bins, map back to continuous AB values

This allows the network to represent multi-modal distributions and commit to specific colors without averaging.

## Dataset

Custom dataset using Pexels API:

- **Size**: 5,800 training / 1,000 test images
- **Resolution**: 512×512 RGB
- **Content**: People, places, natural scenes, objects
- **Preprocessing**: Saturation filtering (threshold = 20/255) to exclude grayscale/washed-out images

## Training Configuration

| Model                | Params | Resolution | Batch Size | Epochs | Loss                   |
| -------------------- | ------ | ---------- | ---------- | ------ | ---------------------- |
| CNN Baseline         | 4.3M   | 96×96      | 512        | 20     | 0.7 MSE + 0.3 L1       |
| U-Net RGB            | 4.5M   | 96×96      | 512        | 20     | 0.7 MSE + 0.3 L1       |
| U-Net LAB            | 20.5M  | 512×512    | 16         | 15     | 0.5 MSE + 0.5 L1       |
| U-Net Classification | 20.5M  | 512×512    | 16         | 20     | Weighted Cross-Entropy |

All models use Adam optimizer with learning rate 1e-4.

## Results

### Quantitative Performance

| Model                | Resolution | PSNR (dB) | Approach       |
| -------------------- | ---------- | --------- | -------------- |
| CNN Baseline         | 96×96      | 11.2      | Regression     |
| U-Net RGB            | 96×96      | 13.5      | Regression     |
| U-Net LAB            | 512×512    | 19.0      | Regression     |
| U-Net Classification | 512×512    | 18.2      | Classification |

**Peak Signal-to-Noise Ratio (PSNR)** measures reconstruction quality:

```
PSNR = 10 × log₁₀(1.0 / MSE)
```

Typical ranges: <20 dB = poor, 20-30 dB = acceptable, >30 dB = good.

**Note**: PSNR correlates with MSE but doesn't always reflect perceptual quality. Two colorizations with identical PSNR may differ significantly in color saturation and semantic correctness.

### Qualitative Analysis

**CNN Baseline**: Heavily desaturated, brownish results with color bleeding. Fine details lost due to bottleneck.

**U-Net RGB**: Clear improvement in saturation and spatial localization. Skip connections preserve edges and reduce bleeding. Some averaging persists in ambiguous regions.

**U-Net LAB**: More vibrant colors and sharper boundaries at 512×512. LAB formulation helps focus on chrominance. Occasionally over-saturates.

**U-Net Classification**: Most saturated and confident colors. Commits to specific choices rather than averaging. More natural and semantically plausible for objects with strong color priors (grass, sky, skin). However, visible quantization artifacts appear as discrete boundaries in smooth gradients.

## Key Findings

**Architecture Impact:**

- Skip connections in U-Net provide 2.3 dB improvement by preserving spatial information
- Multi-scale feature preservation crucial for color boundary sharpness

**Color Space Benefits:**

- LAB formulation reduces complexity by separating luminance from chrominance
- Enables model to focus on color prediction, producing more saturated outputs

**Regression vs Classification:**

- Classification avoids conditional mean convergence by representing full distributions
- Produces more confident, saturated predictions
- Trade-offs: quantization artifacts, class imbalance, higher memory

**Loss Function Effects:**

- MSE alone → smoothest but most desaturated
- L1 → increased saturation but potential artifacts
- Combined (α = 0.5-0.7) → balanced trade-off

## Limitations

1. **Color averaging persists**: Even with combined losses and LAB space, regression models produce desaturated averages in ambiguous regions

2. **Quantization artifacts**: Classification model shows visible discrete boundaries in smooth gradients

3. **Ambiguous objects**: All models struggle with objects lacking strong color priors (walls, synthetic objects, varied-color clothing)

4. **Class imbalance**: Rare colors remain underrepresented despite weighted loss

5. **Metric limitations**: PSNR doesn't fully capture perceptual quality or semantic correctness

## Future Directions

- Perceptual losses from pre-trained networks for semantic correctness
- Interactive user guidance through sparse color hints
- Soft binning schemes or learnable color centers
- Probabilistic sampling from predicted distributions rather than argmax
- Generative models (GANs, diffusion) for more realistic colorizations

## License

Educational purposes. Dataset images sourced from Pexels under their license terms.
