
# 🌨️ C-HR-GAN: Concordance-High-Resolution GAN

> **C-HR-GAN** is an enhanced image-to-image translation framework based on pix2pixHD, tailored for structurally consistent translation under complex conditions such as **snow-covered terrain**.

It integrates:

- **SSIM-based structural constraints**  
- **SE-ResNet blocks with learnable residual weighting**  

These innovations significantly improve **realism** and **geometric fidelity**, making the model especially suitable for scientific remote sensing applications like **rock mass surface detection under snow**.

---

## 📌 Project Highlights

### 🔧 Enhancements over pix2pixHD

#### 1. Structure-Aware GAN Loss (SSIM-Enhanced)

- Introduces a novel loss:

  ```
  L_GAN + λ × (1 - SSIM), where λ = 2
  ```

- Ensures that the generated images are not only visually realistic but also **structurally aligned** with the target.
- Especially effective in snow-covered surface reconstruction tasks.

#### 2. SE-ResNet Residual Blocks

- Replaces standard ResNet blocks with **Squeeze-and-Excitation (SE)** residual blocks.
- Introduces a **learnable residual weight** to adaptively control residual contributions.
- Enhances attention to informative features and fine-grained structures in the scene.

---

## ⚙️ Setup Instructions

### Step 1️⃣: Install PyTorch

Please refer to the [official PyTorch website](https://pytorch.org) for installation instructions.

Example (CUDA 11.3 on Linux):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

### Step 2️⃣: Install Python Dependencies

Run the following command to install all required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Testing Instructions

⚠️ **Important Note:**  
The paired high-resolution SAR-optical dataset used for training is not publicly available due to security restrictions.

If you need access to the **trained model weights (.pth)**, please **contact the corresponding author**.

### 🔧 Preparation

1. Place the trained weights at:

```
./checkpoints/Uragen_SAR-to-optical
```

2. Prepare your test SAR images (both snow-free and snow-covered) under:

```
./datasets/Uragen/test_A/
```

---

### ▶️ Run Testing

Execute the test script:

```bash
python test.py   --name Uragen_SAR-to-optical   --no_instance   --label_nc 0   --resize_or_crop none   --dataroot ./datasets/Uragen
```

---

### 📁 Output Location

Test results will be saved to:

```
./results/Uragen_SAR-to-optical/test_latest/images
```

---

## 📬 Contact

For further questions or access to the model weights, please contact:

**Liming He**  
📧 heliming@mail.neu.edu.cn
