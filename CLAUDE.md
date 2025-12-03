# CLAUDE.md - AI Assistant Guide for google-colab Repository

## Repository Overview

This is a Google Colab notebooks repository focused on **deep learning**, **machine learning**, and **quantitative finance**. The notebooks are primarily educational, covering topics from basic PyTorch operations to advanced transformer architectures and financial derivatives hedging.

**Primary Language**: Python (Jupyter Notebooks)
**Main Frameworks**: PyTorch, NumPy, Pandas, Matplotlib
**Target Environment**: Google Colab with Google Drive integration

## Directory Structure

```
google-colab/
├── data/                      # Financial datasets
│   ├── kdd17/                 # KDD17 stock price dataset
│   │   ├── ourpped/           # Preprocessed stock data (50 tickers)
│   │   ├── price_long_50/     # Long-format price data
│   │   └── trading_dates.csv  # Trading calendar
│   ├── stocknet-dataset/      # Additional stock data
│   └── SPY.csv                # S&P 500 ETF data
├── deep hedging/              # Deep hedging for derivatives
│   ├── deep hedge 1-3.ipynb   # Progressive hedging implementations
│   └── pytorch/               # PyTorch-specific implementations
├── deep learning/             # Core deep learning tutorials
│   ├── PYTORCH/               # PyTorch fundamentals (8+ notebooks)
│   ├── pytorch tencho/        # Korean PyTorch tutorials
│   ├── Transformer.ipynb      # Transformer architecture
│   └── Torch.nn.ipynb         # Neural network modules
├── Plus transformer/          # Advanced transformer models
│   ├── BERT.ipynb             # BERT implementation
│   ├── GPT_2.ipynb            # GPT-2 implementation
│   ├── ViT.ipynb              # Vision Transformer
│   ├── Swin Transformer.ipynb # Swin Transformer
│   └── ConvNeXt.ipynb         # ConvNeXt architecture
├── vola/                      # Volatility modeling
│   ├── garch.ipynb            # GARCH models
│   └── Untitled0-6.ipynb      # Various volatility experiments
└── README.md
```

## Key Technical Areas

### 1. Deep Learning Fundamentals (`deep learning/`)
- **PyTorch basics**: Tensors, autograd, neural network construction
- **Topics covered**:
  - Tensor operations and indexing
  - Automatic differentiation (autograd)
  - Neural network modules (`nn.Linear`, `nn.Sequential`)
  - Weight initialization (He, Xavier)
  - Custom datasets and dataloaders
  - CNN feature maps and classification

### 2. Transformer Architectures (`Plus transformer/`)
- Complete implementations of major transformer models
- **BERT**: Bidirectional encoder representations
- **GPT-2**: Autoregressive language model
- **ViT**: Vision Transformer for image classification
- **Swin Transformer**: Hierarchical vision transformer
- **ConvNeXt**: Modernized ConvNet architecture

### 3. Deep Hedging (`deep hedging/`)
- Financial derivatives hedging using neural networks
- **Key concepts**:
  - Black-Scholes delta hedging simulation
  - Brownian motion stock price simulation
  - Hansen's skewed t-distribution for returns
  - LSTM-based GARCH networks (GARCHNetLSTM)
  - Custom loss functions (negative log-likelihood)

### 4. Volatility Modeling (`vola/`)
- GARCH model implementations
- Time series analysis for financial data

## Data Conventions

### KDD17 Dataset
- **50 stock tickers** with historical OHLCV data
- **Date ranges**: Training ends 2015-01-01, Validation ends 2016-01-01, Test ends 2017-01-01
- **Features computed**: Price ratios, returns, volume changes, moving averages, price trends

### Feature Engineering (from `vola/garch.ipynb`)
```python
# Standard features for stock data:
- open, high, low (as % of close)
- close, adj_close (returns)
- volume_change, log_volume
- volume_ma{5,10,20}, volume_std{5,10,20}
- price_trend{5,10,15,20,25,30}
```

## Code Patterns and Conventions

### PyTorch Model Structure
```python
class ModelName(nn.Module):
    def __init__(self, input_size, hidden_size, ...):
        super().__init__()
        # Define layers

    def forward(self, x):
        # Forward pass
        return output
```

### Google Colab Integration
- Drive mounting: `/content/drive/MyDrive/Colab Notebooks/`
- Data paths often reference: `/content/drive/MyDrive/Colab Notebooks/google-colab/data`

### Training Loop Pattern
```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # Validation
```

## Language Notes

- **Mixed Korean/English**: Many notebooks contain Korean comments and variable names
- Korean tensor dimension names in einops: `'개 단 차'` = batch, sequence, feature
- Some notebooks use Korean for section headers and explanations

## Development Guidelines for AI Assistants

### When Modifying Notebooks
1. Preserve existing cell outputs when possible
2. Maintain consistent import patterns at the top of notebooks
3. Keep Korean comments if present in the original
4. Test code compatibility with Google Colab environment

### When Working with Models
1. Check for GPU availability: `torch.cuda.is_available()`
2. Use `.to(DEVICE)` pattern for model/tensor placement
3. Implement proper gradient clipping for training stability
4. Save model checkpoints with optimizer state

### When Handling Data
1. Verify data paths exist before loading
2. Handle NaN values appropriately in financial data
3. Use proper train/val/test splits based on dates (not random)
4. Normalize features when required

### Common Dependencies
```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from einops import rearrange  # For transformer models
```

## Known Issues and TODOs

1. **GARCHNetLSTM**: Missing `num_stocks` parameter in some training scripts
2. **PerStockAttentionLSTM**: Class referenced but not defined in some notebooks
3. Some notebooks have incomplete implementations (marked with errors in output)

## File Naming Conventions

- Numbered notebooks indicate sequence: `PYTORCH 1.ipynb`, `PYTORCH 2.ipynb`
- `Untitled*.ipynb` files are typically work-in-progress experiments
- Model names in filenames: `ViT.ipynb`, `BERT.ipynb`, `GPT_2.ipynb`

## Useful Commands

```bash
# Check GPU availability in Colab
!nvidia-smi

# Install common packages
!pip install einops torchinfo torchviz

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```
