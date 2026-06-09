# credit-card-fraud-detection
An end-to-end ML pipeline comparing ANN (Focal Loss) and XGBoost for highly imbalanced credit card fraud data.

# Credit Card Fraud Detection: ANN vs. XGBoost Ensemble Pipeline

This repository contains an end-to-end Machine Learning pipeline designed to detect fraudulent credit card transactions. The project tackles severe class imbalance (0.2% fraud cases) using advanced loss functions, custom class weighting, and a robust, leakage-free cross-validation framework.

##  Key Highlights & Engineering Choices
* **Severe Imbalance Mitigation:** Optimized a Deep ANN using `BinaryFocalCrossentropy` alongside custom class weighting, and tuned XGBoost's `scale_pos_weight` to aggressively handle the 0.2% minority class.
* **Leakage-Free Validation Strategy:** Implemented a 5-fold `StratifiedKFold` cross-validation strategy, isolating the `StandardScaler` fitting process strictly to training splits to eliminate data leakage.
* **Dynamic Decision Thresholding:** Instead of using the default 0.5 classification threshold, thresholds were dynamically optimized on validation folds via Precision-Recall curves to maximize the F1-score.
* **Production-Focused Metrics:** Bypassed deceptive ROC-AUC evaluations to optimize strictly for PR-AUC, securing high operational recall (~85-90%) while maintaining functional precision limits to minimize manual review overhead.
* **Ensemble Architecture:** Developed a Soft-Voting Ensemble averaging predicted probability distributions from both models, yielding superior predictive stability.

##  Tech Stack
* **Language:** Python
* **Frameworks/Libraries:** TensorFlow/Keras, XGBoost, Scikit-Learn, Pandas, NumPy
* **Visualization:** Plotly, Seaborn, Matplotlib

## 📊 Pipeline Overview
1. **Data Preprocessing:** Splitting -> Scaling (Fit on Train only, transform on Validation/Test).
2. **Stratified 5-Fold CV:** Training baselines and calculating dynamic thresholds per fold.
3. **Final Training:** Training models on full train split and assessing on an untouched test set.
4. **Ensembling:** Combining outputs via soft-voting for optimized PR-AUC performance.
