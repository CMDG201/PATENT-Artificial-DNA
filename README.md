# METHOD AND SYSTEM FOR IDENTIFYING A MOLECULE OF ARTIFICIAL DEOXYRIBONUCLEIC ACID (DNA) NUCLEOBASE

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=for-the-badge)
![TranSIESTA](https://img.shields.io/badge/TranSIESTA-NEGF%2BDFT-2E86AB?style=for-the-badge)


</div>

---

---

## Overview

It presents a comprehensive investigation into the electrical recognition of eight artificial DNA nucleobases (xDNA and yDNA). By embedding these nucleobases within a nitrogen-terminated graphene nanogap junction, their fingerprint transmission and current readouts were calculated. These readouts serve as a database for training multiple ML models to achieve precise basecalling accuracy. This repository contains the complete **computational datasets**, **Jupyter notebooks**, **optimised hyperparameters**, and **ML model evaluation results**

---

## Repository Structure

```
PATENT-Artifcial-xyDNA/
│
├── README.md                           ← Full technical documentation (this file)
├── xDNA_ATGC.ipynb                     ← xDNA 4-class ML pipeline (Dataset_1)
├── YDNA_ATGC.ipynb                     ← yDNA 4-class ML pipeline (Dataset_2)
├── xyDNABasecalling_C3N_Gap.ipynb      ← 8-class basecalling + robustness tests
├── Datasets.zip                        ← NEGF+DFT T(E) data, all 8 nucleobases
├── xyDNA_Data.zip                      ← Extended rotational + translational dataset
└── xyDNA_Classification.zip            ← All model outputs and classification results
```

---

## Device Architecture and DFT Parameters

### Solid-State In-Plane Nitrogen-Terminated Graphene Nanogap

```
╔══════════════════════════════════════════════════════════════════════╗
║                   SOLID-STATE NANOGAP DEVICE (104)                   ║
║                                                                      ║
║  LEFT ELECTRODE (Source)                   RIGHT ELECTRODE        ║
║  N-terminated graphene        ◄──────►        N-terminated           ║
║                                                graphene (Drain)      ║
║  ████████████████║                          ║████████████████        ║
║  ████████████████║   [xA/xT/xG/xC or       ║████████████████        ║
║  ████████████████║    yA/yT/yG/yC inside]   ║████████████████        ║
║  ████████████████║                          ║████████████████        ║
║                                                                      ║                               ║
╚══════════════════════════════════════════════════════════════════════╝
```
---

## Machine Learning Pipeline
NEGF and DFT based simulations were systematically performed to generate transmission signal datasets for individual nucleobases under multiple structural configurations and operating conditions.
Important numerical characteristics were carefully extracted from the generated signals and then organized into a structured descriptor matrix suitable for advanced machine learning analysis.
Several supervised machine learning algorithms were trained, compared, and hyperparameter optimized in order to identify the most accurate and reliable predictive model.
The highest performing model was further assessed through comprehensive validation procedures using standard statistical performance metrics to confirm robustness and consistency.
The finalized predictive framework was then applied for basecalling tasks to estimate per-nucleobase classification accuracy with interpretable decision outcomes



| Nucleobase | Primary SHAP Driver | Physical Basis |
|-----------|---------------------|---------------|
| xA, xG, xT | **Min** (highest) | Min T(E) encodes anti-resonance/destructive interference unique to each nucleobase junction |
| xC, yT | **Mean** | Mean T(E) distinguishes spectral distribution signatures |
| yA, yC, yT | **Max** | Peak transmission reflects near-Fermi resonance coupling magnitudes |

**Physical insight:**
- **Min for xDNA** → deepest tunnelling suppression points encode junction-specific destructive interference — inaccessible from HOMO/LUMO
- **Max for yDNA** → resonance peak magnitudes at near-Fermi MO energies vary distinctly across yDNA family geometries
- SHAP validates that **T(E)-derived descriptors capture junction-level electronic coupling** — the central inventive claim


### Prerequisites

```bash
pip install numpy pandas scikit-learn shap matplotlib seaborn openpyxl jupyter
```

Recommended: scikit-learn **1.4.2** | Python **3.10** | Google Colab compatible


**© 2025 Indian Institute of Technology Indore. All Rights Reserved.**

</div>
