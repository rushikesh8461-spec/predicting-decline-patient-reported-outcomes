# Predicting a Decline in Patient-Reported Outcomes for Cancer Patients on Chemotherapy

**Course:** UE24CS352A Machine Learning — Mini-Project (Team 8)

## Team Members

| Name | SRN |
|---|---|
| Rushikesh | PES1UG24AM394 |
| Prrajwal | PES1UG24AM386 |

## Project Overview

This project explores whether patient-reported outcomes (PROs) can be used to predict a decline in health-related quality of life during chemotherapy. It is inspired by *Predicting a Decline in Patient Reported Outcomes for Cancer Patients on Chemotherapy*, a Stanford CS229 Spring 2020 project by Ostberg and Peterson.

The project uses machine-learning models to study changes in patient-reported health outcomes using baseline and on-treatment information.

**Data note:** The original Stanford STARR electronic health record data is private. Results obtained using synthetic data or a separately supplied cleaned dataset are experimental and should not be interpreted as validated clinical findings or used for medical decisions.

## Project Structure

```text
main/
├── README.md
├── data/
│   └── breast_pro_clean.csv
├── model/
│   └── ml_experiments.ipynb
└── outfolder/
    └── Generated experiment results
```

- `data/breast_pro_clean.csv` — cleaned dataset used by the notebook.
- `model/ml_experiments.ipynb` — notebook containing the machine-learning experiments.
- `outfolder/` — directory created automatically to store experiment outputs.

## Objectives

- Explore the prediction of declining patient-reported outcomes during chemotherapy.
- Prepare and preprocess the available dataset.
- Compare multiple machine-learning algorithms.
- Evaluate model performance using appropriate validation techniques.
- Investigate the contribution of different feature groups.
- Examine fitted LASSO coefficients for model interpretation.

## Machine-Learning Models

The notebook evaluates the following models:

1. LASSO
2. Ridge Regression
3. Elastic Net
4. K-Nearest Neighbors (KNN)
5. LightGBM
6. Support Vector Machine (SVM)
7. Random Forest
8. Feed-Forward Neural Network (FFNN)
9. Soft-Voting Ensemble

The experiments compare imputation strategies and use nested stratified cross-validation for model tuning and evaluation.

## Methodology

The workflow includes:

1. Loading and checking the dataset.
2. Preprocessing features and handling missing values.
3. Preparing binary prediction outcomes.
4. Training and tuning machine-learning models.
5. Evaluating performance using cross-validation.
6. Comparing model results.
7. Performing optional feature-group ablation and LASSO coefficient analysis.
8. Saving experiment outputs for further analysis.

## Running the Notebook

### Requirements

- Python 3.10 or newer
- pandas
- NumPy
- Matplotlib
- scikit-learn
- joblib
- LightGBM
- Jupyter Notebook

### Install Dependencies

Run this command in your terminal:

```bash
python -m pip install pandas numpy matplotlib scikit-learn joblib lightgbm jupyter
```

### Steps to Run

1. Clone or download this repository.
2. Ensure that `breast_pro_clean.csv` is located in the `data/` directory.
3. Ensure that `ml_experiments.ipynb` is located in the `model/` directory.
4. Start Jupyter Notebook:

   ```bash
   python -m jupyter notebook
   ```

5. Open `model/ml_experiments.ipynb`.
6. Run the notebook cells from top to bottom.
7. Review the generated files in `outfolder/`.

Run the notebook from the project root or the `model/` directory so that its dataset path can be resolved correctly.

## Output Files

Depending on which notebook sections finish successfully, the experiments may generate the following files:

| File | Description |
|---|---|
| `exp1_model_comparison.csv` | Model comparison and AUROC summaries |
| `oof_predictions
