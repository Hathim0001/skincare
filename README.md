# 🌿 Skincare Sage: AI-Powered Dermatological Analysis & Personalized Skincare Recommendation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()

**Skincare Sage** is an intelligent, end-to-end dermatological advisory system that leverages deep transfer learning and vectorized recommendation algorithms to analyze facial skin conditions and generate personalized skincare regimens.

By combining deep convolutional neural networks (**EfficientNet-V2**) for computer vision inference with a **Cosine Similarity** multi-attribute recommendation engine, Skincare Sage bridges the gap between clinical skin assessment and personalized cosmetic dermatology.

---

## 📌 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Features](#-key-features)
- [System Pipeline](#-system-pipeline)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Streamlit Application](#running-the-streamlit-application)
  - [CLI Inference](#cli-inference)
- [Machine Learning & Recommendation Models](#-machine-learning--recommendation-models)
  - [Acne Severity Classifier](#1-acne-severity-classifier)
  - [Skin Type Predictor](#2-skin-type-predictor)
  - [Cosine Similarity Recommender](#3-cosine-similarity-recommender)
- [Datasets & Web Scraping](#-datasets--web-scraping)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🔬 Overview & Architecture

Choosing appropriate skincare products often poses a significant challenge due to misidentified skin types, fluctuating acne severity, and vast catalogs of conflicting product ingredients. 

**Skincare Sage** solves this problem through a structured, multi-stage pipeline:
1. **Visual Diagnostic Inference:** Analyzes facial portrait imagery to classify skin type (*Normal, Dry, Oily*) and diagnose acne severity (*Clear, Mild, Severe, Extreme*).
2. **User Attribute Synergy:** Blends visual diagnostic outputs with user-selected secondary skin concerns (e.g., hydration, dark circles, pore care, anti-aging).
3. **High-Dimensional Vector Matching:** Computes similarity metrics across categorized skincare products (cleansers, moisturizers, sunscreens, serums, face masks) to produce ranked, actionable recommendations.

```mermaid
flowchart TD
    A[User Facial Image] --> B[Image Preprocessing & Normalization 224x224]
    B --> C[EfficientNet-V2 Acne Classifier]
    B --> D[Skin Type Classifier CNN]
    
    C -->|Acne Severity Level| E[Feature Vector Synthesis]
    D -->|Skin Type Profile| E
    F[User Secondary Skin Concerns] --> E
    
    E -->|18-Dimensional Sparse Vector| G[Cosine Similarity Engine]
    H[(Curated Skincare Database)] --> G
    
    G --> I[Categorized Product Recommendations]
    I --> J[Cleansers]
    I --> K[Moisturizers]
    I --> L[Sunscreens & Treatments]
```

---

## ✨ Key Features

- **Automated Acne Severity Classification:** 4 granular clinical tiers (*Clear*, *Mild*, *Severe*, *Extreme*) trained on facial dermatological datasets.
- **Accurate Skin Type Profiling:** Deep learning inference distinguishing *Normal*, *Dry*, and *Oily* skin patterns based on epidermal texture and sheen.
- **18-Attribute Feature Vectorization:** Incorporates clinical skin conditions and secondary consumer concerns including *hydration, dullness, fine lines, wrinkles, pore care, dark circles, sensitivity*, and *daily maintenance*.
- **Multi-Category Regimen Formulation:** Generates balanced routines across vital skincare categories: Cleansers, Face Moisturisers, Sunscreens, Concealers, and Masks.
- **E-Commerce Metadata Integration:** Real-time catalog entries including product brand, price, target skin type, active concerns, a