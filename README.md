# 🌿 PlantVillage Disease Classification

Deep learning model for classifying apple leaf diseases using the PlantVillage dataset.

## 📊 Dataset

**Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

**Classes (4):**
- Apple Scab (630 images)
- Black Rot (621 images)
- Cedar Apple Rust (275 images)
- Healthy (1,645 images)

**Total Images:** 3,171

**Split:**
- Training: 2,219 (70%)
- Validation: 475 (15%)
- Test: 477 (15%)

## 🏗️ Model Architecture

**ConvNeXt-Tiny** - Modern CNN architecture with:
- **Total Parameters:** 27,823,204
- **Trainable Parameters:** 27,823,204
- **Input Size:** 224×224 RGB images
- **Architecture Highlights:**
  - Depthwise convolutions (7×7 kernels)
  - Layer normalization
  - GELU activation
  - Stochastic depth for regularization
  - Inverted bottleneck design (1×4 expansion ratio)

### Layer Structure
```
Stage 1: 96 channels  (3 blocks)
Stage 2: 192 channels (3 blocks)
Stage 3: 384 channels (9 blocks)
Stage 4: 768 channels (3 blocks)
```

## 🎯 Training Approaches

### 1️⃣ From Scratch Training (Current Implementation)

**Configuration:**
- Epochs: 40
- Batch Size: 32
- Learning Rate: 1e-4
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Mixed Precision: Enabled (FP16)

**Results:**
- Final Train Accuracy: 76.16%
- Final Val Accuracy: 76.42%
- Best Val Accuracy: 79.79% (Epoch 27)
- Training Time: ~18.44 minutes

**Pros:**
- Full control over learning
- No dependency on pre-trained weights
- Model learns task-specific features

**Cons:**
- Requires more data
- Longer training time
- Lower accuracy with limited data

### 2️⃣ Transfer Learning (Recommended)

**Why Transfer Learning is Better:**

✅ **Higher Accuracy** - Pre-trained weights from ImageNet provide better feature extraction  
✅ **Faster Convergence** - Typically reaches optimal performance in 10-15 epochs  
✅ **Less Data Required** - Works well even with smaller datasets  
✅ **Better Generalization** - Pre-learned features transfer well to similar tasks  

**Expected Improvements:**
- Accuracy: 85-95% (vs 76% from scratch)
- Training Time: 50% faster
- Stability: More consistent results

## 🔧 Optimization Details

**Optimizer:** AdamW
- Weight decay: 0.01 (default)
- Betas: (0.9, 0.999)
- Epsilon: 1e-8

**Learning Rate Schedule:**
- Constant LR: 1e-4
- No scheduler used (can add ReduceLROnPlateau for better results)

**Regularization:**
- Stochastic Depth (drop path)
- Data Augmentation:
  - Random Resize Crop (224×224)
  - Random Horizontal Flip
  - Normalization (ImageNet stats)

**Mixed Precision Training:**
- Enabled via `torch.cuda.amp`
- Reduces memory usage by ~40%
- Speeds up training by ~2x on modern GPUs

## 📈 Training Metrics

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 1.1241    | 56.87%    | 1.0108   | 60.42%  |
| 10    | 0.8956    | 70.66%    | 0.8459   | 76.42%  |
| 20    | 0.8656    | 73.91%    | 0.8027   | 77.68%  |
| 30    | 0.8398    | 74.85%    | 0.7900   | 79.16%  |
| 40    | 0.8203    | 76.16%    | 0.7945   | 76.42%  |

**Best Model:** Epoch 27 (Val Acc: 79.79%)

## 🚀 Usage

```python
# Load trained model
model = torch.load('best_model.pth')
model.eval()

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
```

## 📁 Project Structure

```
PlantVillage-Disease-Classification/
├── plant_disease_classification.ipynb  # Main training notebook
├── README.md                           # This file
└── models/                             # Saved model checkpoints
    └── best_model.pth
```

## 🔬 Future Improvements

- [ ] Implement transfer learning with pre-trained weights
- [ ] Add learning rate scheduler (CosineAnnealingLR)
- [ ] Experiment with data augmentation (RandAugment, MixUp)
- [ ] Try ensemble methods
- [ ] Add Grad-CAM visualization
- [ ] Deploy as REST API

## 📝 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
pandas
```

## 🎓 References

- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## 📄 License

MIT

---

**Developed by Eşref Batuhan Simsar**
