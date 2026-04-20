# METHOD AND SYSTEM FOR IDENTIFYING A MOLECULE OF ARTIFICIAL DEOXYRIBONUCLEIC ACID (DNA) NUCLEOBASE

<div align="center">

![Patent](https://img.shields.io/badge/Indian%20Patent-202521002474-1F4E79?style=for-the-badge)
![IIT Indore](https://img.shields.io/badge/IIT%20Indore-Applicant-FF6B00?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=for-the-badge)
![TranSIESTA](https://img.shields.io/badge/TranSIESTA-NEGF%2BDFT-2E86AB?style=for-the-badge)
![License](https://img.shields.io/badge/License-Patent%20Protected-DC143C?style=for-the-badge)

### Indian Patent Application No. **202521002474**
**Title:** METHOD AND SYSTEM FOR IDENTIFYING A MOLECULE OF ARTIFICIAL DEOXYRIBONUCLEIC ACID (DNA) NUCLEOBASE

*Applicant: Indian Institute of Technology Indore | Filed: January 10, 2025*

**Inventors:** Milan Kumar Jena · Sneha Mittal · Prof. Biswarup Pathak

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Repository Structure](#-repository-structure)
3. [Scientific Background](#-scientific-background)
4. [Device Architecture and DFT Parameters](#-device-architecture-and-dft-parameters)
5. [Quantum Transport: NEGF+DFT Formalism](#-quantum-transport-negfdft-formalism)
6. [Molecular Orbital Analysis](#-molecular-orbital-analysis)
7. [Machine Learning Pipeline](#-machine-learning-pipeline)
8. [Complete Hyperparameter Tables](#-complete-hyperparameter-tables)
9. [Key Results](#-key-results)
10. [Robustness Under Dynamic Configurations](#-robustness-under-dynamic-configurations)
11. [SHAP Explainability Analysis](#-shap-explainability-analysis)
12. [How to Run](#-how-to-run)
13. [Dataset Description](#-dataset-description)
14. [Distinction from Prior Art D1](#-distinction-from-prior-art-d1-wo2015038972a1)
15. [FER Sufficiency Addendum](#-fer-sufficiency-addendum-section-104)
16. [Patent and Funding Information](#-patent-and-funding-information)
17. [License](#-license-and-legal-notice)

---

## 🔬 Overview

This repository contains the complete **computational datasets**, **Jupyter notebooks**, **optimised hyperparameters**, and **ML model evaluation results** supporting **Indian Patent Application No. 202521002474**.

The invention presents a first-of-its-kind method for precise, single-molecule electric identification of **eight artificially synthesized benzo-homologated DNA nucleobases** — **xDNA** (xA, xT, xG, xC) and **yDNA** (yA, yT, yG, yC) — using:

- A **solid-state in-plane nitrogen-terminated graphene nanogap device**
- **Non-Equilibrium Green's Function + DFT (NEGF+DFT)** quantum transport via TranSIESTA
- **Energy-resolved fingerprint transmission functions T(E)** as primary ML descriptors
- A **Min/Max/Mean normalisation feature engineering pipeline**
- A **Random Forest Classifier (RFC)** with **SHAP explainability**

> ⚠️ **Patent Notice:** All data, code, methods, and results are covered by Indian Patent Application No. 202521002474. Reproduction or commercial use without written authorisation from IIT Indore is strictly prohibited.

---

## 🗂️ Repository Structure

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

## 🧪 Scientific Background

### What are xDNA and yDNA?

Artificial/benzo-homologated/expanded DNA nucleobases are synthetic nucleotides formed by formally fusing a benzene ring with each canonical DNA base, producing two families:

```
Natural base + Benzene ring → xDNA family:  xA, xT, xG, xC
                            → yDNA family:  yA, yT, yG, yC
```

| Property | Natural DNA | xDNA / yDNA |
|----------|------------|-------------|
| Number of bases | 4 | **8** |
| Sequence complexity | 4ⁿ | **8ⁿ** |
| HOMO–LUMO gap | Larger | **Smaller** (molecular nanowire) |
| Thermal stability | Standard | **Higher** (stronger stacking) |
| π-conjugation | Standard | **Extended** |
| Fluorescence | No | **Yes** |

### Why is Identification Challenging?

xDNA/yDNA tunnelling signals exhibit **substantial spectral overlap** due to comparable shape, size, frontier MO energies, and local density of states — especially among pyrimidine members (xT/xC, yT/yC). Conventional HOMO/LUMO descriptors cannot reliably distinguish all eight nucleobases. This is the core problem the invention solves.

---

## ⚙️ Device Architecture and DFT Parameters

### Solid-State In-Plane Nitrogen-Terminated Graphene Nanogap

```
╔══════════════════════════════════════════════════════════════════════╗
║                   SOLID-STATE NANOGAP DEVICE (104)                   ║
║                                                                      ║
║  LEFT ELECTRODE (Source)        14 Å          RIGHT ELECTRODE        ║
║  N-terminated graphene        ◄──────►        N-terminated           ║
║                                                graphene (Drain)      ║
║  ████████████████║                          ║████████████████        ║
║  ████████████████║   [xA/xT/xG/xC or       ║████████████████        ║
║  ████████████████║    yA/yT/yG/yC inside]   ║████████████████        ║
║  ████████████████║                          ║████████████████        ║
║                                                                      ║
║  Supercell: 24.00 × 17.16 × 42.49 Å³                               ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Complete Device and DFT Parameters (from Supporting Info §1.1)

| Parameter | Value |
|-----------|-------|
| Electrode material | Graphene |
| Edge termination | Nitrogen-functionalized |
| Supercell (x, y, z) | 24.00 × 17.16 × 42.49 Å³ |
| **N–N gap distance** | **14 Å** |
| Left / Right electrode | Electron source / Electron drain |
| Geometry optimisation (isolated) | B3LYP / 6-31+G* (Gaussian 09) |
| Nanogap DFT code | SIESTA |
| Exchange-correlation | vdW-DF-cx (van der Waals corrected) |
| Pseudopotentials | Norm-conserving Troullier–Martins |
| Basis set | Double-ζ polarised (DZP) |
| Mesh cutoff | 200 Ry |
| k-points | 1 × 3 × 2 |
| Density matrix convergence | 0.0001 eV |
| Relaxation algorithm | Conjugate Gradient (CG) |
| Force convergence criterion | < 0.01 eV/Å |
| Transport code | TranSIESTA |
| Isosurface (MO plots) | 0.005 e/Å³ |

---

## ⚛️ Quantum Transport: NEGF+DFT Formalism

### 1. Transmission Function T(E) — Eq. S1

$$T(E) = \text{Tr}\left[\Gamma_L(E)\, G_C(E)\, \Gamma_R(E)\, G_C^{\dagger}(E)\right]$$

| Symbol | Definition |
|--------|-----------|
| T(E) | Energy-resolved quantum transport transmission probability |
| Γ_L(E) = i[Σ_L − Σ_L†] | Left electrode broadening matrix |
| Γ_R(E) = i[Σ_R − Σ_R†] | Right electrode broadening matrix |
| G_C(E) = [(E+iη)S − H − Σ_L − Σ_R]⁻¹ | Retarded Green's function (device region) |
| G_C†(E) | Advanced (conjugate transpose) Green's function |

**Energy window:** (E − E_F) = **−2.5 to 0 eV** | Fermi level shifted to zero

**What T(E) captures that HOMO/LUMO cannot:**
- Orbital symmetry matching between electrode and nucleobase MOs
- Resonance broadening and Fano-type interference effects
- Energy-resolved MO delocalization across the full junction
- Electrode–nucleobase electronic coupling at each energy value

### 2. Breit–Wigner Transmission (Chen and Tao, Acc. Chem. Res. 2009)

$$T(E) = \frac{4\Gamma_L \Gamma_R}{(E_F - \varepsilon_{MO})^2 + (\Gamma_L + \Gamma_R)^2}$$

where ε_MO is the MO energy closest to E_F and Γ_L/R are electrode coupling strengths.

### 3. Conductance and Sensitivity

$$G(V_g) = \frac{2e^2}{h} \cdot T(\mu), \qquad \mu = E_F - eV_g$$

$$\text{Conductance Sensitivity (\%)} = \frac{G_x - G_0}{G_0} \times 100$$

Gate voltage for sensitivity evaluation: V_g = −0.865 eV (xDNA), −0.865 eV (yDNA)

**Key result:** xA/xG and yA/yG/yC show distinctly higher conductance sensitivity than pyrimidines.

### 4. Current–Voltage Calculation — Eq. S2

$$I(V_b) = \frac{2e}{h} \int T(E, V_b)\left[f_L(E - \mu_L) - f_R(E - \mu_R)\right] dE$$

I–V range: 0 to 0.2 V | Increment: 0.05 V | Current sensitivity at V = 0.2 V

**Key result:** xG/yG show highest current sensitivity, consistent with conductance results.

---

## 🔭 Molecular Orbital Analysis

### Frontier MO Energies and Transport Behaviour (Supporting Info §2, Figure S4)

| Nucleobase | MO Energy (E−E_F) | MO Localisation | T(E) Signature | Current |
|-----------|-------------------|-----------------|----------------|---------|
| xA | −0.865 eV | Dense, coupled | Sharp, broad peaks | High |
| xG | −0.005 eV | Resonance (near-Fermi) | Amplified T(E) | Elevated |
| xT | −0.035 eV | Asymmetric, feeble | Reduced | Low |
| xC | −0.035 eV | Asymmetric, feeble | Reduced | Low |
| yA | −0.005 eV | Dense, near-Fermi | Sharp peaks | High |
| yG | −1.145 eV | Strong coupling | Amplified | Elevated |
| yT | −2.225 eV | Deep-level; weak | Minimal | Minimal |
| yC | −0.115 eV | Feeble, asymmetric | Reduced | Low |

**Physical mechanism:** Purine nucleobases (xA, xG, yA, yG) have larger surface area enabling stronger hydrogen bonding with N-terminated graphene edges, enhanced π-orbital delocalization, and resonance-coupled T(E) signatures. Pyrimidines show weaker, asymmetric MO localisation.

**Electrode–nucleobase interaction energy:** Purines ~1–2 eV; Pyrimidines ~0.3–0.8 eV (Figure S4a).

---

## 🤖 Machine Learning Pipeline

### Complete Workflow

```
NEGF+DFT → T(E) per nucleobase per config
     ↓
Feature extraction: TF, Min=TF/min(TF), Max=TF/max(TF), Mean=TF/mean(TF)
     ↓
Descriptor matrix: 8000 pts/dataset (500 × 4 descriptors × 4 nucleobases)
     ↓
80/20 train-test split + RandomizedSearchCV hyperparameter tuning
     ↓
7 ML models benchmarked → RFC best (99.50% xDNA, 97.75% yDNA)
     ↓
10-fold cross-validation + learning curve analysis
     ↓
Confusion matrix + ROC-AUC + Precision/Recall/F1
     ↓
SHAP global + local explainability
     ↓
RFC basecalling: 100% unknown pool → per-nucleobase accuracy (%)
```

### Feature Engineering Code

```python
import numpy as np

def extract_features(TF):
    tf_min  = np.min(TF,  axis=1, keepdims=True)
    tf_max  = np.max(TF,  axis=1, keepdims=True)
    tf_mean = np.mean(TF, axis=1, keepdims=True)
    Min_norm  = TF / tf_min
    Max_norm  = TF / tf_max
    Mean_norm = TF / tf_mean
    return np.column_stack([
        np.squeeze(tf_min),
        np.squeeze(np.min(Min_norm,   axis=1)),
        np.squeeze(np.max(Max_norm,   axis=1)),
        np.squeeze(np.mean(Mean_norm, axis=1)),
    ])
# Dataset: 8000 pts per family (500 configs x 4 descriptors x 4 nucleobases)
```

### ML Models Benchmarked (scikit-learn 1.4.2, Python 3.10)

| Model | xDNA Test Acc. | yDNA Test Acc. |
|-------|----------------|----------------|
| **RFC** ⭐ | **99.50%** | **97.75%** |
| DTC | 99.12% | 96.25% |
| GPC | 43.00% | 44.25% |
| KNN | 70.50% | 77.00% |
| SVC | 34.33% | 23.00% |
| LRC | 26.50% | 28.00% |
| GNB | 32.37% | 22.00% |

---

## 📊 Complete Hyperparameter Tables

### RFC — Optimised Parameters (both datasets identical core settings)

```python
RandomForestClassifier(
    n_estimators        = 100,
    criterion           = 'entropy',
    max_depth           = 25,
    max_features        = 'sqrt',
    min_samples_leaf    = 1,
    min_samples_split   = 2,
    bootstrap           = True,
    oob_score           = False,
    ccp_alpha           = 0.0,
    random_state        = 100
)
# xDNA: Train 100.00% | Test 99.50%
# yDNA: Train 99.88%  | Test 97.75%
```

### All Models — xDNA Dataset (Table S1)

| Model | Key Hyperparameters | Train | Test |
|-------|---------------------|-------|------|
| KNN | n_neighbors=30, metric='minkowski', leaf_size=30, weights='uniform' | 78.25% | 70.50% |
| **RFC** | criterion='entropy', max_depth=25, n_estimators=100, max_features='sqrt', random_state=100 | **100%** | **99.50%** |
| SVC | C=1.0, kernel='rbf', gamma='scale', degree=3, tol=0.001 | 35.81% | 34.33% |
| DTC | criterion='entropy', max_depth=23, random_state=65, splitter='best' | 99.99% | 99.12% |
| LRC | C=1.0, penalty='l2', solver='lbfgs', max_iter=100, random_state=35 | 24.62% | 26.50% |
| GNB | var_smoothing=1e-09 | 34.25% | 32.37% |
| GPC | kernel=None, multi_class='one_vs_rest', optimizer='fmin_l_bfgs_b' | 98.56% | 43.00% |

### All Models — yDNA Dataset (Table S2)

| Model | Key Hyperparameters | Train | Test |
|-------|---------------------|-------|------|
| KNN | n_neighbors=45, metric='minkowski', leaf_size=60, weights='uniform' | 75.00% | 77.00% |
| **RFC** | criterion='entropy', max_depth=25, n_estimators=100, max_features='sqrt', random_state=100 | **99.88%** | **97.75%** |
| SVC | C=1.0, kernel='rbf', gamma='scale', tol=0.001 | 26.44% | 23.00% |
| DTC | criterion='gini', max_depth=23, random_state=65, splitter='best' | 99.94% | 96.25% |
| LRC | C=1.0, penalty='l2', solver='lbfgs', max_iter=100, random_state=35 | 25.31% | 28.00% |
| GNB | var_smoothing=1e-09 | 25.50% | 22.00% |
| GPC | kernel=None, multi_class='one_vs_rest', optimizer='fmin_l_bfgs_b' | 97.56% | 44.25% |

### 10-Fold Cross-Validation — RFC (Figure S6)

Both xDNA and yDNA: ~99% and ~98% mean CV score respectively across all 10 folds.
Training and cross-validation scores converge at N > 1500 data points, confirming no overfitting.

---

## 📈 Key Results

### Quaternary Classification (4-class, per family)

| Dataset | Accuracy | AUC (min) |
|---------|----------|-----------|
| xDNA (xA,xT,xG,xC) | **99.50%** | 0.91 |
| yDNA (yA,yT,yG,yC) | **97.75%** | 0.91 |

### Predictive Basecalling (100% unknown pool)

| Nucleobase | Accuracy | Miscalling | Miscalled As |
|-----------|---------|-----------|-------------|
| **xA** | **100.00%** | 0.00% | — |
| **xT** | **100.00%** | 0.00% | — |
| xG | 99.80% | 0.20% | xT |
| **xC** | **100.00%** | 0.00% | — |
| yA | 99.80% | 0.20% | yC |
| yT | ~98.20% | ~1.80% | Mixed |
| yG | 99.60% | 0.40% | Mixed |
| yC | 99.80% | 0.20% | yT |

### Eight-Class Unified Classification

| Nucleobase | Precision | Sensitivity | F1 Score |
|-----------|-----------|-------------|----------|
| xA | 1.00 | 1.00 | 1.00 |
| xC | 0.98 | 1.00 | 0.99 |
| xG | 1.00 | 0.99 | 0.99 |
| xT | 0.99 | 0.98 | 0.99 |
| yA | 0.98 | 1.00 | 0.99 |
| yC | 1.00 | 0.98 | 0.99 |
| yG | 0.98 | 0.99 | 0.98 |
| yT | 1.00 | 0.99 | 1.00 |
| **Overall** | | **99.12%** | |

### Pair Classification Summary

| Pair | Accuracy | Dominant Descriptor |
|------|---------|---------------------|
| xA/xT | **100%** | Max (82%) |
| xG/xC | **100%** | Min (49%) |
| yA/yT | **99.50%** | Max (72%) |
| yG/yC | **99.50%** | Max (82%) |
| xA/xG | **100%** | Min (76%) |
| yA/yG | **99.75%** | Min (34%) + Max (36%) |
| xT/xC | **99.50%** | Min (43%) |
| yT/yC | **99.75%** | Min (57%) |

---

## 🔄 Robustness Under Dynamic Configurations

### Rotational Dynamics (Accuracy% | Miscalling%)

| Nucleobase | 0° | 60° | 120° | 180° |
|-----------|-----|-----|------|------|
| xA | 100\|0 | 98.5\|1.5 | 100\|0 | 100\|0 |
| xT | 100\|0 | 100\|0 | 98.2\|1.8 | 99.8\|0.2 |
| xG | 99.8\|0.2 | 99.5\|0.5 | 99.8\|0.2 | 98.2\|1.8 |
| xC | 100\|0 | 99.8\|0.2 | 99.8\|0.2 | 99.8\|0.2 |
| yA | 99.8\|0.2 | 98.5\|1.5 | 99.5\|0.5 | 98.6\|1.4 |
| yT | 98.2\|1.8 | 99.5\|1.5 | 99.6\|0.4 | 98.8\|1.2 |
| yG | 99.6\|0.4 | 98.2\|1.8 | 99.8\|0.2 | 96.5\|3.5 |
| yC | 99.8\|0.2 | 98.5\|1.5 | 96.5\|3.5 | 99.8\|0.2 |

### Translation (Inplane ±0.5 Å | Outplane ±1.0 Å)

| Nucleobase | In +0.5Å | In −0.5Å | Out +1.0Å | Out −1.0Å |
|-----------|---------|---------|---------|---------|
| xA | 99.2\|0.8 | 97.0\|3.0 | 98.5\|1.5 | 96.2\|3.8 |
| xT | 99.8\|0.2 | 99.5\|0.5 | 96.2\|3.8 | 96.8\|3.2 |
| xG | 99.2\|0.8 | 98.6\|1.4 | 97.5\|2.5 | 98.6\|1.4 |
| xC | 99.8\|0.2 | 99.8\|0.2 | 99.8\|0.2 | 99.8\|0.2 |
| yA | 98.6\|1.4 | 97.5\|2.5 | 98.2\|1.8 | 96.8\|3.2 |
| yT | 99.5\|0.5 | 99.6\|0.4 | 97.5\|2.5 | 97.6\|2.4 |
| yG | 95.4\|4.5 | 96.2\|3.8 | 96.8\|3.2 | 94.0\|6.0 |
| yC | 97.5\|2.5 | 99.2\|0.8 | 96.5\|3.5 | 95.5\|4.5 |

Rotation has minimal impact (>98% for most). Outplane translation most challenging — yG minimum ~94% at ±1.0 Å. Noise mitigation: train ML on bare nanogap data (no nucleotide) as noise class.

---

## 🔍 SHAP Explainability Analysis

### Global Feature Importance (RFC + SHAP consistent)

| Dataset | Rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---------|--------|--------|--------|--------|
| **xDNA** | **Min** 63% | Mean 16% | Max 12% | TF 9% |
| **yDNA** | **Max** 44% | TF 40% | Min 10% | Mean 6% |
| **8-class** | **Min** 60% | Mean 17% | Max 16% | TF 7% |

### Per-Nucleobase Local SHAP (Figure 4c,d)

| Nucleobase | Primary SHAP Driver | Physical Basis |
|-----------|---------------------|---------------|
| xA, xG, xT | **Min** (highest) | Min T(E) encodes anti-resonance/destructive interference unique to each nucleobase junction |
| xC, yT | **Mean** | Mean T(E) distinguishes spectral distribution signatures |
| yA, yC, yT | **Max** | Peak transmission reflects near-Fermi resonance coupling magnitudes |

**Physical insight:**
- **Min for xDNA** → deepest tunnelling suppression points encode junction-specific destructive interference — inaccessible from HOMO/LUMO
- **Max for yDNA** → resonance peak magnitudes at near-Fermi MO energies vary distinctly across yDNA family geometries
- SHAP validates that **T(E)-derived descriptors capture junction-level electronic coupling** — the central inventive claim

---

## 🚀 How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn shap matplotlib seaborn openpyxl jupyter
```

Recommended: scikit-learn **1.4.2** | Python **3.10** | Google Colab compatible

### Complete RFC Training and SHAP Example

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import shap

def extract_features(TF):
    tf_min  = np.min(TF,  axis=1, keepdims=True)
    tf_max  = np.max(TF,  axis=1, keepdims=True)
    tf_mean = np.mean(TF, axis=1, keepdims=True)
    return np.column_stack([
        np.squeeze(tf_min),
        np.squeeze(np.min(TF / tf_min, axis=1)),
        np.squeeze(np.max(TF / tf_max, axis=1)),
        np.squeeze(np.mean(TF / tf_mean, axis=1)),
    ])

X = np.vstack([extract_features(xA_TF), extract_features(xT_TF),
               extract_features(xG_TF), extract_features(xC_TF)])
y = np.array([0]*500 + [1]*500 + [2]*500 + [3]*500)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rfc = RandomForestClassifier(
    n_estimators=100, criterion='entropy', max_depth=25,
    max_features='sqrt', min_samples_leaf=1, min_samples_split=2,
    bootstrap=True, oob_score=False, random_state=100)
rfc.fit(X_train, y_train)

print(f"Test Accuracy: {accuracy_score(y_test, rfc.predict(X_test))*100:.2f}%")
print(classification_report(y_test, rfc.predict(X_test),
      target_names=['xA','xT','xG','xC']))

cv = cross_val_score(rfc, X, y, cv=10)
print(f"10-fold CV: {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")

explainer   = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test,
    feature_names=['TF','Min','Max','Mean'],
    class_names=['xA','xT','xG','xC'], plot_type='bar')
```

---

## 📁 Dataset Description

### `Datasets.zip` — Equilibrium T(E) Data

| File | Nucleobase | Shape | Energy Range |
|------|-----------|-------|-------------|
| xA_TF.csv | xA | 500 × 500 | (E−E_F) = −2.5 to 0 eV |
| xT_TF.csv | xT | 500 × 500 | same |
| xG_TF.csv | xG | 500 × 500 | same |
| xC_TF.csv | xC | 500 × 500 | same |
| yA_TF.csv | yA | 500 × 500 | same |
| yT_TF.csv | yT | 500 × 500 | same |
| yG_TF.csv | yG | 500 × 500 | same |
| yC_TF.csv | yC | 500 × 500 | same |

Total ML dataset per family: **8000 data points** (500 configs × 4 descriptors × 4 nucleobases)

### `xyDNA_Data.zip` — Dynamic Configurations

Rotational (0°, 60°, 120°, 180°) + inplane (±0.5 Å) + outplane (±1.0 Å) T(E) for all 8 nucleobases.

### `xyDNA_Classification.zip` — All Model Outputs

Confusion matrices, ROC-AUC curves, SHAP plots, learning curves, pair classification results.

---

## 🆚 Distinction from Prior Art D1 (WO2015038972A1)

| Dimension | D1 — QM-Seq (WO2015038972A1) | Present Invention (202521002474) |
|-----------|------------------------------|----------------------------------|
| **Device type** | STM-STS: Pt-Ir tip over Au(111) gold — vertical probe-over-substrate | Solid-state **in-plane** planar nanogap |
| **Electrode** | Au(111) gold. Graphene = generic alternative, unfunctionalised | Graphene with **N-functionalized edges** — both electrodes |
| **Gap** | Dynamic tip–substrate distance (piezo-controlled); no fixed parameter | Fixed **N–N gap = 14 Å** |
| **Targets** | Natural DNA/RNA (A,G,C,T,U) + chemically modified natural bases | **Exclusively xDNA and yDNA** (8 named expanded nucleobases) |
| **Observable** | Experimental I–V; scalar HOMO/LUMO from dI/dV | NEGF+DFT **T(E)=Tr[Γ_L G_C Γ_R G_C†]** — junction-level, energy-resolved |
| **Framework** | DFT/B3LYP for isolated MO only. **No NEGF. No TranSIESTA.** | Full NEGF+DFT via TranSIESTA; complete electrode–molecule–electrode modelled |
| **Features** | Generic "parameter extraction" — no pipeline, no normalisation | **Min/Max/Mean T(E) normalisation** → 4D feature matrix; 8000 pts/dataset |
| **ML** | Concept only; no workflow, no dataset, no trained model | 7 models benchmarked; RFC trained (Table S1/S2), 10-fold CV |
| **Explainability** | None | **SHAP** — quantitative, per-nucleobase, per-descriptor |
| **Accuracy** | Not demonstrated for xDNA/yDNA | **100%** (xA, xT, xC); **99.12%** (8-class unified) |

---

## 📋 FER Sufficiency Addendum [Section 10(4)]

### Objection (a) — RFC Architectural Parameters

✅ **Resolved.** Complete parameters: `criterion='entropy'`, `max_depth=25`, `n_estimators=100`, `max_features='sqrt'`, `random_state=100`. Implemented in `xDNA_ATGC.ipynb` and `YDNA_ATGC.ipynb`. Stability confirmed by 10-fold CV and learning curve convergence at N > 1500 data points (Figure S6).

### Objection (b) — Mathematical Technique for T(E) → Descriptors

✅ **Resolved.** NEGF equation T(E) = Tr[Γ_L G_C Γ_R G_C†] defined in Supporting Info Eq. S1. Min/Max/Mean normalisation fully implemented in notebooks. SHAP analysis provides quantitative physical validation — normalised descriptors encode electronic coupling and MO delocalization physics directly relevant to the claims.

### Objection (c) — No Working Example or Experimental Data

✅ **Resolved.** Three executable Jupyter notebooks constitute complete working examples. `Datasets.zip` contains all NEGF+DFT T(E) data. `xyDNA_Classification.zip` contains all numerical results — confusion matrices, AUC curves, precision/recall/F1 scores — for all eight nucleobases across all classification scenarios.

---

## 📋 Patent and Funding Information

| Field | Details |
|-------|---------|
| Application Number | 202521002474 |
| Date of Filing | January 10, 2025 |
| Date of Publication | September 26, 2025 |
| First Examination Report | March 06, 2026 |
| **FER Response Due** | **September 06, 2026** |
| Controller | Dr. Md Jawed Ansaree |
| Examiner | Ayush Poonia |
| Patent Agent | Manisha Singh, LexOrbis, New Delhi |

### Inventors

| Name | Affiliation | ORCID |
|------|------------|-------|
| Milan Kumar Jena | Dept. Chemistry, IIT Indore | 0000-0002-8363-1291 |
| Sneha Mittal | Dept. Chemistry, IIT Indore | 0000-0003-2567-4274 |
| Prof. Biswarup Pathak | Dept. Chemistry, IIT Indore | 0000-0002-9972-9947 |

### Funding

| Agency | Project Number |
|--------|----------------|
| BRNS | 2023-BRNS/12356 |
| SPARC | CRG/2022/000836 |
| CSIR | 01(3046)/21/EMR-II |
| MoE | Fellowship (M.K.J.) |
| UGC | Fellowship (S.M.) |
| IIT Indore | HPC + Laboratory |

---

## ⚖️ License and Legal Notice

This repository is published solely to support **Indian Patent Application No. 202521002474** as working computational examples demonstrating enablement under Section 10(4) of the Patents Act, 1970. All content is the intellectual property of **Indian Institute of Technology Indore**.

**Any reproduction, commercial use, or incorporation without written authorisation from IIT Indore is strictly prohibited.**

Scientific enquiries: **Prof. Biswarup Pathak** | biswarup@iiti.ac.in  
Patent enquiries: **LexOrbis** | mail@lexorbis.com

---

<div align="center">

*Indian Patent Application No. 202521002474 | IIT Indore | Filed: January 10, 2025*

**© 2025 Indian Institute of Technology Indore. All Rights Reserved.**

</div>
