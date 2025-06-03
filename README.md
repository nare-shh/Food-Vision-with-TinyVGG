# Vision Transformer (ViT) Food Classification Notebook

This notebook implements a Vision Transformer (ViT) model for food image classification, specifically trained to classify images of pizza, steak, and sushi. The implementation follows the architecture described in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).

## Features

- Implementation of Vision Transformer (ViT) architecture from scratch
- Training on food image dataset (pizza, steak, sushi)
- Visualization of model architecture and training process
- Model evaluation and prediction capabilities
- Support for both custom and pretrained ViT models

## Requirements

```python
torch>=1.12.0
torchvision>=0.13.0
matplotlib
torchinfo
requests
```

## Notebook Structure

1. **Setup and Imports**
   - Environment setup
   - Required library imports
   - Version checks for PyTorch and torchvision

2. **Data Preparation**
   - Download and setup of food image dataset
   - Data transformation pipeline
   - DataLoader creation

3. **Model Architecture**
   - Patch Embedding implementation
   - Position Embedding
   - Multi-Head Self-Attention
   - Transformer Encoder Blocks
   - MLP Blocks
   - Complete ViT model

4. **Training Process**
   - Model initialization
   - Loss function and optimizer setup
   - Training loop implementation
   - Evaluation metrics
   - Loss curve visualization

5. **Prediction and Visualization**
   - Model prediction on new images
   - Visualization of results
   - Performance analysis

## Key Components

### Patch Embedding
```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        # Implementation details
```

### Multi-Head Self-Attention
```python
class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0):
        # Implementation details
```

### Transformer Encoder
```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072):
        # Implementation details
```

## Usage

1. **Setup Environment**
   ```python
   # Check and install required versions
   try:
       import torch
       import torchvision
       assert int(torch.__version__.split(".")[1]) >= 12
       assert int(torchvision.__version__.split(".")[1]) >= 13
   except:
       !pip3 install -U torch torchvision torchaudio
   ```

2. **Download and Prepare Data**
   ```python
   # Download food image dataset
   image_path = download_data(
       source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
       destination="pizza_steak_sushi"
   )
   ```

3. **Create and Train Model**
   ```python
   # Initialize model
   vit = ViT(num_classes=len(class_names))
   
   # Train model
   results = train(
       model=vit,
       train_dataloader=train_dataloader,
       test_dataloader=test_dataloader,
       optimizer=optimizer,
       loss_fn=loss_fn,
       epochs=10,
       device=device
   )
   ```

## Model Architecture Details

The Vision Transformer architecture consists of:

1. **Image Patching**
   - Input images are split into fixed-size patches (16x16)
   - Each patch is flattened and projected to a fixed dimension

2. **Position Embedding**
   - Learnable position embeddings are added to patch embeddings
   - Class token is prepended to the sequence

3. **Transformer Encoder**
   - Multiple layers of self-attention and MLP blocks
   - Layer normalization and residual connections
   - GELU activation functions

4. **Classification Head**
   - Final layer normalization
   - Linear projection to number of classes

## Training Parameters

- Learning rate: 3e-3
- Batch size: 32
- Number of epochs: 10
- Optimizer: Adam with weight decay
- Loss function: Cross Entropy Loss

## Visualization

The notebook includes various visualizations:
- Training and validation loss curves
- Training and validation accuracy curves
- Sample predictions with confidence scores
- Model architecture visualization

## Notes

- The implementation uses the ViT-Base configuration by default
- The model can be trained on any image classification dataset by modifying the data loading section
- GPU acceleration is automatically used if available
- The notebook includes both custom implementation and pretrained model options
