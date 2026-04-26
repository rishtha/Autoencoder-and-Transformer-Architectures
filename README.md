# Autoencoder and Transformer Architectures

Deep learning assignment implementing autoencoder-based anomaly detection 
and transformer architectures from scratch using PyTorch.  
**Course:** CSE 676-B — Deep Learning, University at Buffalo (Spring 2025)

## Overview
This project covers four core deep learning implementations:
- **Part I:** Theoretical analysis of autoencoder and self-attention mechanisms
- **Part II:** Autoencoder for anomaly detection on real-world time-series data
- **Part III:** Transformer model built from scratch for text classification
- **Part IV:** Fine-tuning BART (facebook/bart-base) for abstractive summarization

## Tech Stack
- **Language:** Python
- **Framework:** PyTorch
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, 
  HuggingFace Transformers, torchinfo, sacrebleu, bert-score
- **Tools:** Jupyter Notebook, TensorBoard

## Project Structure
├── a2_part_2.ipynb        # Autoencoder anomaly detection
├── a2_part_3.ipynb        # Transformer from scratch
├── a2_part_4.ipynb        # BART fine-tuning for summarization
├── a2_part_1.pdf          # Theoretical derivations
└── a2_weights.txt         # Link to saved model weights (UBbox)

## Key Results
| Part | Task | Metric | Result |
|------|------|--------|--------|
| II | Anomaly Detection | Test Accuracy | >80% |
| III | Text Classification | Test Accuracy | >80% |
| IV | Summarization (BART) | ROUGE-1 | >40 |

## How to Run
```bash
git clone https://github.com/rishtha/Autoencoder-and-Transformer-Architectures.git
cd Autoencoder-and-Transformer-Architectures
pip install -r requirements.txt
jupyter notebook
```

## Topics Covered
- Autoencoder architecture design and anomaly detection thresholding
- Variational Autoencoders (VAE)
- Self-attention and scaled dot-product attention
- Transformer encoder/decoder implementation from scratch
- Positional encoding and multi-head attention
- LLM fine-tuning with HuggingFace (BART)
- Evaluation: ROUGE, BLEU, BERTScore, Precision, Recall, F1

## Author
Rishitha Saravanan Priya  
[LinkedIn](https://linkedin.com/in/rishithasp) | [Portfolio](https://rishitha.dev)
