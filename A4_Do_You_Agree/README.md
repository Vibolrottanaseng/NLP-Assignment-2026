# A4 - Do you Agree ?

This assignment implements a custom BERT model from scratch and applies Sentence-BERT (SBERT) architecture for Natural Language Inference (NLI). The system learns semantic relationships between sentence pairs and provides a web-based interface for testing text similarity and inference predictions.

## Assignment Overview

The objective of this assignment is to:

1. Implement BERT from scratch based on Transformer architecture.
2. Train Sentence-BERT using siamese networks to generate semantic sentence embeddings.
3. Perform Natural Language Inference (NLI) classification.
4. Develop a web application to demonstrate model capability.

---

## Task 1 — BERT from Scratch

### Dataset: 

Source: Bookcorpus [https://huggingface.co/datasets/rojagtap/bookcorpus]



A custom implementation of Bidirectional Encoder Representations from Transformers (BERT) was developed using file name BERT-update.ipynb

### Key Components:

- Token embedding
- Positional embedding
- Segment embedding
- Multi-head self-attention
- Feed-forward networks
- Layer normalization

### Training Objective:

- Masked Language Modeling (MLM)

The model was trained on a subset of publicly available datasets.

---
## Task 2 — Sentence-BERT (SBERT)

### Dataset

Source: SNLI  [https://huggingface.co/datasets/stanfordnlp/snli]

The pretrained BERT encoder was adapted into a Siamese Network structure.

### Architecture:

- Shared BERT encoder for premise and hypothesis
- Sentence embedding generation
- Feature combination
- Softmax classification head

### Output Classes:

- Entailment
- Neutral
- Contradiction

---

## Task 3 — Evaluation

The model performance was evaluated using classification metrics:

- Precision
- Recall
- F1-score
- Accuracy

Example metrics:

| Class | Precision | Recall | F1-score | Support |
|------|-----------|--------|---------|-------|
| Entailment | 0.00 | 0.00 | 0.00 | 3329
| Neutral | 0.00 | 0.00 | 0.00 | 3235
| Contradiction | 0.33 | 1.00 | 0.50 | 3278

---

## Task 4 — Web Application

A simple web interface was developed. 

### Features:

Two input fields:

  - Premise
  - Hypothesis
  - Predict NLI relationship between sentences
  - Displays predicted label






