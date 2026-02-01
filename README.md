# CSE 256 Programming Assignment 1: Sentiment Analysis with Neural Networks

**Author:** Siddhant Hitesh Mantri  
**PID:** A69041429

This repository contains implementations for sentiment analysis using Deep Averaging Networks (DAN) with various configurations, including Bag-of-Words baselines, GloVe embeddings, random embeddings, and Byte Pair Encoding (BPE) tokenization.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Files](#data-files)
- [Running the Code](#running-the-code)
  - [Part 1: DAN with GloVe Embeddings](#part-1-dan-with-glove-embeddings)
  - [Part 1b: DAN with Random Embeddings](#part-1b-dan-with-random-embeddings)
  - [Part 2: BPE Tokenization](#part-2-bpe-tokenization)
  - [Baseline: Bag-of-Words](#baseline-bag-of-words)
- [Expected Outputs](#expected-outputs)
- [Results Directory](#results-directory)
- [Notes](#notes)

---

## Project Structure

```
CSE256_PA1_WI26/
│
├── main.py                              # Main script to run all experiments
├── BOWmodels.py                         # Bag-of-Words baseline models
├── DANmodels.py                         # Deep Averaging Network implementations
├── BPEmodels.py                         # BPE-based DAN models
├── BPE.py                               # Byte Pair Encoding implementation
├── sentiment_data.py                    # Data loading utilities
├── utils.py                             # Helper functions
├── README.md                            # This file
│
├── data/
│   ├── train.txt                        # Training data
│   ├── dev.txt                          # Development/validation data
│   ├── glove.6B.50d-relativized.txt    # 50-dimensional GloVe embeddings
│   └── glove.6B.300d-relativized.txt   # 300-dimensional GloVe embeddings
│
└── results/                             # Generated plots and saved models
    ├── *.png                            # Various visualization plots
    └── bpe_*.pkl                        # Saved BPE models
```

---

## Requirements

This project requires Python 3.7+ and the following packages:

```
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
```

---

## Installation

1. **Clone or download the repository:**
   ```bash
   cd CSE256_PA1_WI26
   ```

2. **Install required packages:**
   ```bash
   pip install torch numpy matplotlib scikit-learn
   ```

3. **Verify installation:**
   ```bash
   python3 -c "import torch; import numpy; import matplotlib; print('All dependencies installed successfully!')"
   ```

---

## Data Files

Ensure the following data files are present in the `data/` directory:

- `train.txt`: Training dataset with sentiment labels
- `dev.txt`: Development/validation dataset
- `glove.6B.50d-relativized.txt`: 50-dimensional GloVe word embeddings
- `glove.6B.300d-relativized.txt`: 300-dimensional GloVe word embeddings

**Note:** The GloVe embedding files are preprocessed and relativized for this assignment.

---

## Running the Code

All experiments are run through the `main.py` script with different model arguments. The script will automatically create a `results/` directory to store output plots and saved models.

### Part 1: DAN with GloVe Embeddings

This runs a comprehensive grid search over 16 configurations with pretrained GloVe embeddings:
- **Embedding dimensions:** 50d, 300d
- **Hidden layer sizes:** 100, 256
- **Number of layers:** 2, 3
- **Dropout positions:** embedding layer, hidden layer

**Command:**
```bash
python3 main.py --model DAN
```

**Training time:** Approximately 20-30 minutes (varies by system)

**What it does:**
1. Loads GloVe embeddings (50d and 300d)
2. Trains 16 different DAN configurations (100 epochs each)
3. Reports best development accuracy for each configuration
4. Generates multiple visualization plots (see [Expected Outputs](#expected-outputs))

**Output:**
- Console output with progress for each configuration
- Summary table showing all configurations and their best/final dev accuracies
- 9 plots saved in `results/` directory

---

### Part 1b: DAN with Random Embeddings

This trains a DAN model with randomly initialized embeddings (no pretrained GloVe) using the best configuration from Part 1.

**Command:**
```bash
python3 main.py --model DANRANDOM
```

**Training time:** Approximately 2-3 minutes

**Configuration:**
- Embedding dimension: 300d (randomly initialized)
- Hidden layer size: 256
- Number of layers: 2
- Dropout: 0.3 at embedding layer

**What it does:**
1. Creates a vocabulary from training data
2. Initializes random embeddings
3. Trains DAN for 100 epochs
4. Compares performance with pretrained embeddings

**Output:**
- Console output with training progress
- Best and final dev accuracies
- 1 comprehensive analysis plot: `results/dan_random_emb_comprehensive.png`

---

### Part 2: BPE Tokenization

This experiments with Byte Pair Encoding tokenization at different vocabulary sizes.

**Command:**
```bash
python3 main.py --model BPE
```

**Training time:** Approximately 15-25 minutes

**Vocabulary sizes tested:**
- 500 merges (~600 tokens)
- 1,000 merges (~1,100 tokens)
- 2,000 merges (~2,100 tokens)
- 5,000 merges (~5,100 tokens)
- 10,000 merges (~10,100 tokens)

**What it does:**
1. Trains or loads BPE models for each vocabulary size
2. Tokenizes data using learned BPE merges
3. Trains DAN with random embeddings (20 epochs per configuration)
4. Analyzes vocabulary growth and sequence length reduction
5. Compares BPE performance with word-level tokenization

**Output:**
- Console output showing training progress for each vocabulary size
- Saved BPE models: `results/bpe_500.pkl`, `results/bpe_1000.pkl`, etc.
- Analysis plots:
  - `results/bpe_all_configs.png`: Training curves for all vocabulary sizes
  - `results/bpe_vocab_analysis.png`: Vocabulary growth and sequence length analysis

**Note:** BPE models are cached. If you want to retrain, delete the corresponding `.pkl` files in `results/`.

---

### Baseline: Bag-of-Words

**(Optional - Not required for main assignment)**

This runs baseline experiments with simple Bag-of-Words representations.

**Command:**
```bash
python3 main.py --model BOW
```

**What it does:**
1. Creates BoW representations with 512-dimensional CountVectorizer
2. Trains 2-layer and 3-layer neural networks
3. Compares baseline performance

**Output:**
- `train_accuracy.png`: Training accuracy comparison
- `dev_accuracy.png`: Dev accuracy comparison

---

## Expected Outputs

### Part 1: DAN with GloVe Embeddings

**Console Output:**
```
Using device: cpu
============================================================
Loading 50d embeddings...
============================================================
Embeddings loaded in: 2.34 seconds
--- Config: 50D_100H_2Layers_EmbDropout ---
Embedding: 50d, Hidden: 100, Layers: 2, Dropout: 0.3 (embedding)
Epoch #10: train acc 0.893, dev acc 0.756, train loss 0.234, dev loss 0.567
...
Best dev accuracy: 0.779
...
======================================================================
GRID SEARCH RESULTS SUMMARY
======================================================================
Config                         Best Dev Acc    Final Dev Acc  
------------------------------------------------------------
50D_100H_2Layers_EmbDropout    0.779           0.768
50D_100H_2Layers_HidDropout    0.780           0.772
...
300D_256H_2Layers_EmbDropout   0.826           0.819
------------------------------------------------------------
Best configuration: 300D_256H_2Layers_EmbDropout with dev accuracy: 0.826
```

**Generated Plots:**
1. `dan_train_accuracy_50d.png`: Training curves for 50d configs
2. `dan_dev_accuracy_50d.png`: Dev accuracy for 50d configs
3. `dan_train_accuracy_300d.png`: Training curves for 300d configs
4. `dan_dev_accuracy_300d.png`: Dev accuracy for 300d configs
5. `dan_50d_vs_300d_comparison.png`: Direct comparison
6. `dan_best_dev_accuracy.png`: Bar chart of all configurations
7. `dan_loss_curves_50d.png`: Loss curves for 50d
8. `dan_loss_curves_300d.png`: Loss curves for 300d
9. `dan_best_config_detailed.png`: Detailed analysis of best configuration

### Part 1b: DAN with Random Embeddings

**Console Output:**
```
Using device: cpu
============================================================
Part 1b: DAN with Randomly Initialized Embeddings
Config: Embedding: 300d, Hidden: 256, Layers: 2, Dropout: 0.3 (embedding)
============================================================
Vocabulary size: 11088
Data loaded in: 0.23 seconds
Epoch #10: train acc 0.875, dev acc 0.745, train loss 0.289, dev loss 0.612
...
Best dev accuracy: 0.788
Final dev accuracy: 0.784
```

**Generated Plots:**
1. `dan_random_emb_comprehensive.png`: 4-panel comprehensive analysis
   - Accuracy over time (train vs dev)
   - Loss over time (train vs dev)
   - Overfitting gap
   - Loss difference

### Part 2: BPE Tokenization

**Console Output:**
```
================================================================================
PART 2: BYTE PAIR ENCODING (BPE) EXPERIMENTS
================================================================================
================================================================================
TRAINING BPE WITH 500 MERGES (APPROX VOCAB SIZE: 600)
================================================================================
Training BPE on train.txt with 500 merges...
Actual vocabulary size: 6296
...
Epoch #20: train acc 0.981, dev acc 0.774, train loss 0.089, dev loss 0.523
Best dev accuracy: 0.774
Average sequence length: 30.4
...
```

**Generated Plots:**
1. `bpe_all_configs.png`: Dev accuracy curves for all vocabulary sizes
2. `bpe_vocab_analysis.png`: Vocabulary growth and sequence length trends

**Saved Models:**
- `bpe_500.pkl`, `bpe_1000.pkl`, `bpe_2000.pkl`, `bpe_5000.pkl`, `bpe_10000.pkl`

---

## Results Directory

After running experiments, the `results/` directory will contain:

```
results/
├── dan_train_accuracy_50d.png
├── dan_dev_accuracy_50d.png
├── dan_train_accuracy_300d.png
├── dan_dev_accuracy_300d.png
├── dan_50d_vs_300d_comparison.png
├── dan_best_dev_accuracy.png
├── dan_loss_curves_50d.png
├── dan_loss_curves_300d.png
├── dan_best_config_detailed.png
├── dan_random_emb_comprehensive.png
├── bpe_all_configs.png
├── bpe_vocab_analysis.png
├── bpe_500.pkl
├── bpe_1000.pkl
├── bpe_2000.pkl
├── bpe_5000.pkl
└── bpe_10000.pkl
```

---

## Notes

### Training Time Estimates
- **Part 1 (DAN):** ~20-30 minutes (16 configurations × ~2 minutes each)
- **Part 1b (Random):** ~2-3 minutes (1 configuration)
- **Part 2 (BPE):** ~15-25 minutes (5 vocabulary sizes × ~3-5 minutes each)

### GPU Support
The code automatically detects and uses GPU if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

To verify GPU usage, check the console output at the start of training.

### Memory Requirements
- Minimum: 8GB RAM
- Recommended: 16GB RAM (especially for 300d embeddings)

### Reproducibility
Results may vary slightly due to random initialization. For reproducible results, set random seeds:
```python
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
```

### Debugging
If you encounter errors:
1. Check that all data files are in the `data/` directory
2. Verify Python version is 3.7+
3. Ensure all dependencies are installed
4. Check available memory (especially for 300d embeddings)

### Quick Test
To verify everything works, run a quick test:
```bash
# Test imports
python3 -c "from BOWmodels import *; from DANmodels import *; from BPEmodels import *; print('All imports successful!')"

# Test data loading
python3 sentiment_data.py
```

---
