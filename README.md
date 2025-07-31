# C-HR-GAN

> **Concordance-High-Resolution generative adversarial network (C-HR-GAN)**

C-HR-GAN is an enhanced version of pix2pixHD, designed for structurally Concordance-High-Resolution generative adversarial network image-to-image translation under challenging conditions such as snow cover terrain. It integrates **SSIM-based structural constraints** and **SE-ResNet blocks with learnable residual weighting**, yielding improved realism and geometric fidelity, especially for scientific and remote sensing applications.

---

## 📌 Project Overview

This project builds on [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), introducing two major enhancements:

### ✅ Key Improvements over pix2pixHD:

1. **Structure-Aware GAN Loss (SSIM-enhanced)**:

   - A novel loss formulation combining adversarial loss with structural similarity index (SSIM):
     > `L_GAN + λ × (1 - SSIM)` (λ=2)
   - Ensures the generated image is not only realistic but structurally aligned with the target (especially important for snow-covered rock mass surface detection).

2. **SE-ResNet Residual Blocks**:

   - Replaces standard ResNet blocks with **Squeeze-and-Excitation (SE)** based residual blocks.
   - Introduces **learnable residual weight** for dynamic control of residual contribution.
   - Enhances attention to informative features and structural regions.

---

## 💻 Prerequisites

Step 1: Install PyTorch
Please follow the official instructions to install PyTorch and its dependencies from:

👉 https://pytorch.org

For example (CUDA 11.3, Linux):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

Step 2: Install Python libraries
Install all required Python libraries using:

pip install -r requirements.txt
```
---


## 🚀 Testing



Due to security concerns, the paired high-resolution SAR-optical image dataset used during the training of the C-HR-GAN model is not publicly available. However, we provide the trained model weights (`.pth` file) located at:

```
./checkpoints/Uragen_SAR-to-optical
```

These can be used to test the performance of the C-HR-GAN model.

To perform testing, simply run the following script:

To test the model:

```bash
python test.py --name Uragen_SAR-to-optical --no_instance --label_nc 0 --resize_or_crop none --dataroot ./datasets/Uragen
```

The test results will be saved to:

```bash
./results/Uragen_SAR-to-optical/test_latest/Image
```

This will translate SAR images (both snow-free and snow-covered) into optical images using the trained model.

* The test SAR images are stored in:

  ```
  ./datasets/Uragen/test_A/
  ```

  * Snow-free SAR images: `S1` to `S3`
  * Snow-covered SAR images: `S4` to `S6`

* The corresponding ground-truth optical images are stored in:

  ```
  ./datasets/Uragen/test_B/
  ```

  * Optical images: `O1` to `O6`

Additionally, we provide 3 optical images under snow-covered conditions (`O4_snow` to `O6_snow`) as references to illustrate the actual degree of snow occlusion in the original scenes.






## 🧪 Model Architecture Summary

### Generator:

- Based on pix2pixHD global generator
- Enhanced with **SE-ResNet blocks**

### Discriminator:

- Multi-scale PatchGAN as in pix2pixHD

### Losses:

- Adversarial Loss (LSGAN or BCE)
- Feature Matching Loss
- VGG Perceptual Loss (optional)
- **SSIM Structural Loss** ✅

---


## 🙏 Acknowledgements

This project is built upon [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) by NVIDIA. We thank the authors for their excellent work.

---

## 🔗 Links

- [pix2pixHD (base)](https://github.com/NVIDIA/pix2pixHD)
- [SSIM paper](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)

---

📧 For questions, please contact: Liming He

