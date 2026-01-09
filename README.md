# GAN-variety
# GAN-Based Data Augmentation for Imbalanced Text Classification

This project investigates the use of Generative Adversarial Networks (GANs) to address class imbalance in a text classification task.

## Implemented Models
- Vanilla GAN (from scratch)
- Conditional GAN (CGAN)
- Wasserstein GAN (WGAN – adapted implementation)

## Pipeline
1. Text preprocessing using TF-IDF
2. Minority-class sample generation using GANs
3. Dataset balancing
4. Classification using MLP
5. Evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix

## Tools
- Python
- PyTorch
- scikit-learn
- Matplotlib / Seaborn

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook



## 📦 requirements.txt
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter

```

# REPORT (PDF) 



## 📝 Report Structure 

### 1. Problem Statement
- Class imbalance problem
- Why it hurts classifiers
- Why GANs are suitable

### 2. Dataset Description & Imbalance Analysis
- Dataset source (Kaggle)
- Number of samples
- Classes
- Imbalance ratios
- Bar chart


### 3. GAN Architectures & Training 

#### 3.1 Vanilla GAN
- Generator & Discriminator
- Trained on minority class only
- Loss: Binary Cross-Entropy

#### 3.2 Conditional GAN (CGAN)
- Conditioning on class labels
- Advantage over Vanilla GAN

#### 3.3 WGAN (Adapted)
- Wasserstein distance
- Critic instead of discriminator
- Weight clipping
- Improved stability


### 4. Classifier Setup & Evaluation 
- MLP architecture
- Training parameters
- Three training scenarios:
  1. Imbalanced dataset
  2. Vanilla GAN balanced
  3. CGAN / WGAN balanced

### 5. Results & Comparisons 
- Table comparing metrics (Accuracy, Precision, Recall, F1)
- Confusion matrices
- Discussion of improvements

### 6. Observations & Conclusions
- GANs improve minority recall
- CGAN / WGAN outperform Vanilla GAN
- Trade-offs (training complexity vs performance)
