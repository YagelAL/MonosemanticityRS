# MonosemanticityRS

A research project investigating **monosemanticity in Recommender Systems** using the Amazon Fashion dataset. The project explores whether Sparse Autoencoders (SAE) and Matryoshka Sparse Autoencoders (MSAE) can learn single, human-interpretable semantic concepts per neuron — improving transparency and interpretability of neural recommendation models.

## Overview

Modern recommender systems based on neural networks are often "black boxes" — individual neurons learn mixed or uninterpretable representations. This project applies **Sparse Autoencoders** on top of a Matrix Factorization (MF) model to encourage monosemantic neuron activations, then labels each neuron using LLM-assisted interpretation.

**Key contributions:**
- Build and train a Matrix Factorization model (K=20 latent factors) on the Amazon Fashion dataset
- Train a Sparse Autoencoder (SAE, 50 hidden neurons) on MF user embeddings
- Train a Matryoshka Sparse Autoencoder (MSAE) with multi-scale prefix training for richer representations
- Label all 50 neurons with semantically meaningful fashion categories using an LLM
- Compare SAE vs. MSAE in terms of recommendation quality and neuron interpretability

## Results

| Metric | SAE | MSAE |
|---|---|---|
| AUC | 0.7685 | 0.7754 |
| Log Loss | 0.5918 | 0.5910 |
| Accuracy | 0.7121 | 0.7209 |
| Precision@10 | 0.0011 | 0.0012 |
| Recall@10 | 0.0057 | 0.0059 |

The MSAE achieves modest but consistent improvements over the plain SAE across all metrics, while both yield 50 interpretable neurons labeled with distinct fashion categories (e.g., *"Women's apparel & accessories"*, *"Socks, underwear & basics"*).

## Repository Structure

```
MonosemanticityRS/
├── Notebooks/                              # Main Jupyter notebooks
│   ├── MF_Notebook.ipynb                   # Matrix Factorization analysis
│   ├── MF_Training_and_Data_Pipeline.ipynb # Data pipeline & MF training
│   └── MSAE_SAE_Amazon.ipynb               # SAE & MSAE training and comparison
│
├── EDA_and_Matrix_Creation_Amazon/         # Exploratory data analysis
│   ├── EDA_and_Matrix_Creation.ipynb       # Build user-item interaction matrix
│   ├── user_item_matrix.npz                # Sparse matrix (400K users × 182K items)
│   └── Mapping Pickles/                    # user/item/category index mappings
│
├── MSAE neuron labeling/                   # Neuron interpretation
│   ├── MSAE_Neuron_Labeling_Notebook.ipynb # LLM-based neuron labeling workflow
│   └── llm_results_msae50_25_items_refined_label/
│       └── amazon_fashion_msae/            # 50 individual neuron label JSON files
│
├── MF_and_MSAE_Training/                   # Training artifacts & visualizations
│   ├── all_neuron_labels.json              # Labels for all 50 neurons
│   ├── mf_model.pkl                        # Trained MF model
│   └── *.png                               # Training & analysis visualizations
│
├── models/                                 # Saved PyTorch model weights
│   ├── sae_model.pth                       # Trained SAE
│   └── mat_sae.pth                         # Trained Matryoshka SAE
│
├── sae_training_11572697.txt               # SAE training log (30 epochs)
└── matryoshka_training_11572697.txt        # MSAE training log (30 epochs)
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch (CPU or GPU)

### Install dependencies

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib plotly
```

### Dataset

The project uses the [Amazon Fashion](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) dataset (ratings/reviews). Download the raw data and run `EDA_and_Matrix_Creation_Amazon/EDA_and_Matrix_Creation.ipynb` to generate the sparse user-item interaction matrix and mapping pickle files.

## Usage

Run the notebooks in the following order:

1. **`EDA_and_Matrix_Creation_Amazon/EDA_and_Matrix_Creation.ipynb`**  
   Loads the Amazon Fashion dataset, builds the sparse user-item interaction matrix (400K users × 182K items), and creates category mappings.

2. **`Notebooks/MF_Training_and_Data_Pipeline.ipynb`**  
   Prepares the training/test split with positive and negative sampling, and trains the Matrix Factorization model.

3. **`Notebooks/MF_Notebook.ipynb`**  
   Detailed analysis of the trained MF model and its latent representations.

4. **`Notebooks/MSAE_SAE_Amazon.ipynb`**  
   Trains both the SAE and the Matryoshka SAE on MF user embeddings, compares recommendation quality metrics, and extracts top-activating items per neuron.

5. **`MSAE neuron labeling/MSAE_Neuron_Labeling_Notebook.ipynb`**  
   Uses an LLM to generate semantic labels for each of the 50 learned neurons based on their top-activating products.

## Model Architecture

### Matrix Factorization (MF)
- 400K users × 182K items, K=20 latent factors
- Trained with BPR-style positive/negative sampling
- Learning rate: 0.005, L2 regularization: 0.01

### Sparse Autoencoder (SAE)
- Input: 20-dimensional MF user embedding
- Hidden layer: 50 neurons with ReLU activation + L1 sparsity penalty, KL divergence and top K sparsity
- Trained for 30 epochs

### Matryoshka Sparse Autoencoder (MSAE)
- Same architecture as SAE but trained with multi-scale prefix loss
- Prefixes: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] neurons
- Encourages hierarchical organization of learned features

## Key Concepts

**Monosemanticity** — A neuron is monosemantic if it activates for a single, coherent semantic concept rather than a mixture of unrelated concepts. In this project, monosemanticity is assessed by examining which product categories most strongly activate each neuron and labeling it with an LLM.

**Sparse Autoencoders (SAE)** — An autoencoder with a bottleneck layer penalized for sparsity (L1 loss). Sparsity encourages each input to activate only a small subset of neurons, helping each neuron specialize.

**Matryoshka SAE (MSAE)** — A variant that applies the reconstruction loss at multiple sub-prefix sizes of the hidden layer simultaneously, inspired by Matryoshka Representation Learning. This produces a nested hierarchy of increasingly detailed representations.

## License

This project is released for research purposes. See the repository for license details.
