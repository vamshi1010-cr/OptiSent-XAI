# OptiSent-XAI: Explainable Multi-Class E-Commerce Product Review Triage, Aspect-Aware Sentiment Calibration, and Concept Drift Monitoring

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-2ea44f?style=for-the-badge&logo=github)](https://vamshi1010-cr.github.io/OptiSent-XAI/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> 🌐 **Live Interactive Observatory**: [https://vamshi1010-cr.github.io/OptiSent-XAI/](https://vamshi1010-cr.github.io/OptiSent-XAI/)

**Course**: `22ITI02 – Industrial / Rural Internship – II` (B.E. IT V Semester, 2026–27)  
**Institute**: Chaitanya Bharathi Institute of Technology (Autonomous), Hyderabad  
**Student**: K Vamshi Chandra Reddy (`160124737111`, Section `IT-2`)  
**Internship Organization**: Navodita Infotech, Pune (`Intern ID: 10110`, `Certificate ID: CNI2606698`)  
**Internal Mentor**: Dr. P. Ramesh Babu, Associate Professor, Dept. of IT, CBIT  
**Head of Department**: Dr. M. Venu Gopalachari, Dept. of IT, CBIT  

---

## 1. Executive Summary & Core Contribution

**OptiSent-XAI** solves the critical industrial vulnerability of conventional sentiment classifiers deployed in modern e-commerce catalogs. Traditional models (including the initial Navodita Infotech Task 3 baseline: 3-class Multinomial Naive Bayes / Logistic Regression on 270 synthetic rows) evaluate reviews strictly on bag-of-words unigram polarity and buyer star ratings. Consequently, they are completely blinded by **5-Star Sarcasm Traps** (e.g., *"5 stars so people see this!! Total trash smart plug melted my wall socket and sparked flames!!"*), misrouting severe safety defects into live catalog publication (`C0 Positive`).

OptiSent-XAI introduces:
1. **Rating-Sentiment Discrepancy Index (RSDI)**: An aspect-calibrated algebraic metric computing the normalized divergence between numerical star ratings and lexical subword polarity:
   $$\text{RSDI} = \left| \frac{\text{stars} - 3.0}{2.0} - \tanh(\text{lexical polarity}) \right|$$
2. **38-Dimensional Hybrid Feature Space**: 24 orthogonal latent semantic components extracted via `TruncatedSVD` over 800 word n-grams `(1,2)` and 1,200 character subword n-grams `(3,5)`, fused with 11 behavioral telemetry features (`star_rating`, `verified_purchase`, `review_velocity_burst`, `helpfulness_ratio`, `char_entropy`, `word_count`, `contrast_marker_count`, `safety_hazard_kw`, `defect_kw`, `praise_kw`, `rsdi_score`) and 3 non-linear interaction terms.
3. **Calibrated Stacking Ensemble**: 40-trial Bayesian TPE-optimized Gradient Boosted Decision Trees (`0.65`) stacked with Calibrated Subword LinearSVC (`0.35`), achieving **91.46% Accuracy**, **90.89% Macro F1**, **0.9491 ROC-AUC**, and **3.674 ms CPU Latency** on held-out test data (`N = 960`).
4. **Automated Concept Drift Defense**: Dual-window `PSI` (Population Stability Index) and `EWMA` (Exponentially Weighted Moving Average, $\lambda = 0.35$) monitoring that intercepts flash-sale lexical surges at Week 6 (`PSI = 0.294 ≥ 0.25`), triggering automated subword retraining that restores Macro F1 from `82.09%` back to `90.94%`.

---

## 2. Repository Structure

```text
OptiSent-XAI-Project/
├── start_demo.bat                          # One-click Windows launch script (FastAPI kernel + browser)
├── start_demo.sh                           # One-command Linux / macOS / WSL2 launcher
├── index.html                              # Kinetic Visual Observatory (Root deployment for GitHub Pages / Vercel)
├── dashboard/
│   └── OptiSent_XAI_Live_Dashboard.html    # Standalone replica for seminar / viva demonstrations
├── backend/
│   ├── __init__.py                         # Backend Python package initializer
│   ├── model.py                            # Production OptiSent-XAI inference engine, RSDI & TreeSHAP
│   └── server.py                           # FastAPI streaming & REST inference server (http://127.0.0.1:8000)
├── optisent_xai_pipeline.py                 # End-to-end Python ML training, TPE optimization & drift pipeline
├── data/
│   └── ecommerce_aspect_triage_4800.csv     # 4,800-row stratified 4-class e-commerce review corpus (16 categories)
├── results/
│   ├── benchmark_results.csv                # 6-model held-out test set benchmark metrics (N = 960)
│   ├── per_class_metrics.csv                # Per-class precision, recall, F1-score & support
│   ├── tpe_optimization_trials.csv          # 40-trial Bayesian TPE 3-fold CV optimization trace
│   ├── feature_attributions.csv             # Global permutation feature importance rankings (ΔF1 drop)
│   └── drift_monitoring_log.csv             # 8-week production streaming PSI, EWMA & KS drift log
├── figures/
│   ├── fig_2_1_architecture.png             # System architecture & hybrid feature fusion diagram
│   ├── fig_2_2_dataset_dist.png             # 4,800-sample category & class distribution
│   ├── fig_3_1_tpe_optimization.png         # 40-trial Bayesian TPE response surface convergence
│   ├── fig_3_2_model_comparison.png         # 6-model benchmark comparison chart
│   ├── fig_3_3_confusion_matrix.png         # Held-out test set confusion matrix (N = 960)
│   ├── fig_3_4_shap_attributions.png        # Global permutation & TreeSHAP feature attribution plot
│   └── fig_3_5_drift_monitoring.png         # 8-week PSI + EWMA concept drift monitoring trajectory
└── requirements.txt                         # Zero-conflict Python dependencies
```

---

## 3. Verified Benchmark Metrics (HARD CONSTRAINTS)

All metrics match the finalized 45-page CBIT Internship Report (`160124737111_K_Vamshi_Chandra_Reddy_Internship_Report.pdf`) and 21-slide Presentation (`160124737111_K_Vamshi_Chandra_Reddy_Internship_PPT.pptx`):

### A. 6-Model Held-Out Test Benchmark (`N = 960`, 80:20 Stratified Split)

| # | Model Architecture | Accuracy | Precision | Recall | Weighted F1 | Macro F1 | ROC-AUC | Log-Loss | Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Multinomial Naive Bayes (Word TF-IDF, N=270) | 70.94% | 72.91% | 70.94% | 67.97% | 65.73% | 0.8923 | 1.2543 | 0.181 ms |
| 2 | Logistic Regression (Word TF-IDF, N=270) | 69.79% | 60.33% | 69.79% | 62.66% | 54.93% | 0.8944 | 1.0685 | 0.172 ms |
| 3 | Calibrated LinearSVC (Word+Char Subword) | 77.40% | 78.45% | 77.40% | 77.01% | 75.02% | 0.9174 | 0.8706 | 0.369 ms |
| 4 | Random Forest (Hybrid SVD + Telemetry) | 88.02% | 88.13% | 88.02% | 87.84% | 87.05% | 0.9428 | 0.4994 | 2.707 ms |
| 5 | Default GBDT (Hybrid SVD + Telemetry) | 89.69% | 89.75% | 89.69% | 89.63% | 88.91% | 0.9473 | 0.4978 | 3.214 ms |
| **6** | **Proposed OptiSent-XAI (TPE-GBDT + Stacking)** | **91.46%** | **91.50%** | **91.46%** | **91.40%** | **90.89%** | **0.9491** | **0.4194** | **3.674 ms** |

*Note: Models 1–3 represent the initial Navodita Infotech Task 3 portal submission (`sentiment_analysis.py` on `product_reviews_sentiment.csv`).*

### B. Per-Class Test Performance (OptiSent-XAI, `878 / 960` Correct)

| Class ID & Designation | Test Support | Precision | Recall | F1-Score | Confusion Matrix Row | Operational SLA Target |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **C0: Positive Endorsement** | 368 (40.0%) | 90.21% | 95.11% | 92.59% | `[350, 10, 6, 2]` | Auto-Publish to Live Product Catalog |
| **C1: Neutral / Mixed-Aspect** | 250 (25.0%) | 92.37% | 87.20% | 89.71% | `[22, 218, 6, 4]` | P2 Product UX Digest (24h SLA) |
| **C2: Negative Quality Defect** | 211 (22.5%) | 92.20% | 95.26% | 93.71% | `[4, 3, 201, 3]` | P1 Vendor QA Ticket (<4h SLA) |
| **C3: Critical Safety / Fraud** | 131 (12.5%) | 92.37% | 83.21% | 87.55% | `[12, 5, 5, 109]` | P0 Trust & Safety SKU Halt (<15m SLA) |

### C. Bayesian TPE Optimization & Feature Permutation Attributions
- **Optimal Trial `#27`**: `93.68%` CV Macro F1 (`learning_rate = 0.0825`, `max_depth = 6`, `l2_regularization = 1.1472`, `min_samples_leaf = 20`, stacking weights `0.65 TPE-GBDT / 0.35 Calibrated LinearSVC`).
- **Global Permutation Importance ($\Delta\text{F1}$ Drop)**: Defect Keyword Density (`+14.45%`), Star Rating (`+8.95%`), Praise Keyword Density (`+5.09%`), Safety Hazard KW (`+4.87%`), Contrastive Markers (`+1.69%`), Helpfulness Ratio (`+0.94%`), Velocity Burst (`+0.91%`).

---

## 4. How to Launch & Demonstrate

### Option 1: One-Click Local Demonstration (FastAPI Backend + Browser)
- **Windows**: Double-click `start_demo.bat`
- **Linux / WSL2 / macOS**: Run `./start_demo.sh`

This automatically starts `uvicorn backend.server:app --port 8000` and opens `http://127.0.0.1:8000` in your default browser.

### Option 2: Standalone Offline Mode (Zero Server Required)
Simply double-click `index.html` or `dashboard/OptiSent_XAI_Live_Dashboard.html` in Chrome, Edge, or Brave. The embedded pure JavaScript inference engine mirrors 100% of the Python ML pipeline, enabling flawless offline presentations or deployment to **GitHub Pages / Vercel**.

---

## 5. Live FastAPI Endpoints (`http://127.0.0.1:8000`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive Kinetic Visual Observatory |
| `GET` | `/api/health` | Verifies kernel status, version 2.1.0, and CBIT student metadata |
| `POST` | `/api/predict` | Computes real-time C0–C3 probabilities, exact RSDI, subwords, TreeSHAP, and legacy MNB comparison |
| `GET` | `/api/stream` | Server-Sent Events (SSE) streaming realistic incoming reviews from the 4,800-row dataset |
| `GET` | `/api/benchmark` | Returns the locked 6-model held-out test metrics and per-class metrics |
| `GET` | `/api/drift` | Returns the 8-week production concept drift trajectory |
| `GET` | `/api/attributions` | Returns verified global permutation feature importance values |
| `GET` | `/docs` | Interactive Swagger UI API documentation |

---

## 6. Three Interactive Visual Acts in the Observatory

1. **Act I — Isometric Parcel Conveyor & Honeybee Optical Scanner**:
   - 60fps physics conveyor with particle wakes trailing moving parcels.
   - Glowing subword trigram callouts (`[melt]`, `[spar]`, `[flam]`, `[swel]`) float upward as parcels cross the rotating hexagonal honeybee scanner.
   - Moving your cursor over the canvas smoothly enters **Slow-Motion Inspection Mode** (0.25x speed), rendering target crosshairs and a live telemetry HUD card.
   - Soundless visual shockwaves pulse outward whenever a **5★ Sarcastic Hazard (`RSDI = 0.892`)** is intercepted and deflected into the **C3 P0 Safety Halt** lane.
2. **Act II — 5-Star Trojan Mask X-Ray & Magnetic Balance Scale**:
   - Interactive **Crack-Open Slider (0% → 100%)**: Drag or slide to crack open the golden 5★ mask shell, exposing the internal hazard coils, sparks, and hidden subword trigrams.
   - Coupled in real-time to the **Magnetic TreeSHAP Balance Beam Canvas**, pulling the beam down from `C0 Positive` (where the initial 270-row baseline fails) into `C3 Safety Halt`.
3. **Act III — Real-Time Brass Gyroscope & 8-Week Concept Drift Wave Studio**:
   - Live 60fps mechanical brass gyroscope with concentric gimbal rings rotating with wobble amplitude directly proportional to the week's **PSI** (`0.042` calm in Week 1 ➔ `0.294` violent wobble in Week 6).
   - Toggling **Self-Healing Retrain** animates a golden harmonic shockwave that dampens the mechanical wobble to zero and locks the streaming distribution waves back into phase for Weeks 7–8 (`90.94% Macro F1`).
