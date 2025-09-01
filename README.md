# Enhanced-Diabetes-Prediction-using-Ensemble-Learning-and-SparsePCA-based-Feature-Reduction-

Overview
The project implements a medical ML workflow: preprocessing with clinically motivated feature dropping, robust scaling, SparsePCA dimensionality reduction, model training (RF, XGBoost, LightGBM, CatBoost, AdaBoost, SVC, Decision Tree), Optuna hyperparameter tuning for top learners, and a stacking meta-learner (logistic regression) that achieves superior accuracy and F1 than any single model on held-out test data. The included CSV offers numeric risk factors and labs (e.g., BMI, FastingBloodSugar, HbA1c, lipids, BP) plus symptoms and medication indicators, with target column Diagnosis for supervised learning.

Key results
Best single tuned learners: boosting family (AdaBoost, CatBoost, LightGBM) after Optuna; tuned AdaBoost notably improves over default.

Final stacked ensemble achieves the highest accuracy (94.68%) and F1 (92.86%) on the test set, outperforming all individual models, including tuned ones, confirming the value of ensemble learning for clinical prediction tasks.

Repository structure
data/diabetes_data.csv: Anonymized dataset with 46 columns; target is Diagnosis (0/1).

paper/Enhanced-Diabetes-Prediction-using-Ensemble-Learning-and-SparsePCA-based-Feature-Reduction-fina.docx: Full research paper detailing methods, experiments, and results, including figures and tables.

notebooks/ or src/: Code for preprocessing, SparsePCA, baseline models, Optuna tuning, and stacking (as described in the paper; align code modules accordingly).

Dataset schema
Size: 1,879 rows, 46 columns; examples include PatientID, Age, BMI, FastingBloodSugar, HbA1c, lipids, blood pressure, symptoms, meds, and Diagnosis (target).

Target balance (Diagnosis): 0 → 1,127; 1 → 752, indicating moderate class imbalance suitable for metrics beyond accuracy (e.g., F1).

Datatypes: mix of int64 and float64 for clinical and lifestyle features; DoctorInCharge is text and should be excluded from modeling.

End-to-end pipeline
Feature selection: Drop non-clinical/bias-prone fields (e.g., PatientID, SocioeconomicStatus, EducationLevel, DoctorInCharge) as described in the paper to keep clinically meaningful signal.

Scaling: Robust Scaler to temper outlier effects common in medical measurements (e.g., extreme glucose or triglycerides).

Dimensionality reduction: SparsePCA to 35 components to speed training, reduce overfitting, and preserve interpretability via sparse loadings.

Modeling: Train RF, XGBoost, LightGBM, CatBoost, AdaBoost, SVC, Decision Tree as baselines for comparative analysis.

Hyperparameter tuning: Optuna for AdaBoost, LightGBM, CatBoost (e.g., learning rate, estimators, depth, regularization) with CV to select robust configurations.

Stacking: Combine top two learners via logistic regression meta-learner with CV folds to avoid leakage and overfitting; evaluate on held-out test set.

Reproducible setup
Split: 80/20 train-test with fixed random seed for reproducibility, matching the methodology in the paper.

Metrics: Report Accuracy and F1 on the test set; F1 is emphasized due to class imbalance and clinical costs of false negatives/positives.

Suggested environment: Python with scikit-learn, Optuna, XGBoost, LightGBM, CatBoost; use RobustScaler and SparsePCA from scikit-learn, and StackingClassifier or custom stacking with logistic regression.

How to run
Step 1: Install dependencies (scikit-learn, optuna, xgboost, lightgbm, catboost); ensure the CSV is available at data/diabetes_data.csv.

Step 2: Preprocess: drop non-clinical IDs/administrative fields, split features/labels, Robust scale numeric features.

Step 3: Fit SparsePCA (n_components ≈ 35) on training set and transform train/test.

Step 4: Train baseline models and record test Accuracy/F1 for comparison and ablations (with/without SparsePCA).

Step 5: Run Optuna studies for AdaBoost, LightGBM, CatBoost with 3-fold CV and ~40 trials; save best params and refit on full training set.

Step 6: Build stacking ensemble using the two best tuned base models and logistic regression meta-learner; evaluate on test set to reproduce ≈94.68% Accuracy and ≈92.86% F1.

Notes on clinical validity and ethics
Feature choices and removals aim to reduce bias and leakage; avoid non-clinical administrative fields in predictive features.

This model is a research prototype; not a medical device; predictions must not replace clinical judgment and should be validated prospectively before real-world use.

Paper linkage
The provided DOCX details the literature survey, methodology, ablations, tuned vs baseline comparisons, and stacked ensemble gains, aligning one-to-one with the code path above for transparent reproducibility and interpretation in a clinical ML context.

Figures in the paper illustrate class distribution, gender stratification, and model metric comparisons; replicate these plots from the evaluation scripts to validate the pipeline end-to-end.

Quick data peek (from CSV)
Head (sample): includes Age, BMI, FastingBloodSugar, HbA1c, BP, lipids, symptoms, meds; Diagnosis present in all rows and used as y.

Types: mostly numeric float/int; ensure categorical-like binary flags remain numeric for tree models and are properly scaled if used in linear/meta models.

Why SparsePCA + stacking
SparsePCA compresses feature space while retaining interpretable loadings, reducing overfitting risk and training time in boosted and linear meta-learners.

Stacking leverages complementary error profiles of tuned boosters, delivering consistent gains on held-out data and the top overall Accuracy/F1 in this study.

Citations and credits
Dataset: anonymized clinical-style diabetes dataset with numerous risk, lab, symptom, and treatment features; target Diagnosis provided for supervised learning.

Research methodology, metrics, and reported results: Enhanced Diabetes Prediction using Ensemble Learning and SparsePCA-based Feature Reduction (DOCX in paper/)
