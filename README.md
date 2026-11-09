# Heart Disease Risk Prediction with a Safety Layer

A machine learning project that predicts heart disease risk using **Logistic Regression** and adds a **confidence-based safety layer**: when the model is not confident, it does not decide, it refers the case to a doctor.

>  **Disclaimer:** This is a learning project, **not a medical device**. It must not be used for real medical diagnosis or treatment decisions.

##  Project Overview
In medical screening, **missing a sick patient (False Negative)** is far more dangerous than a false alarm. So this project goes beyond accuracy:
1. Build a clean, leakage-free ML pipeline
2. Compare Logistic Regression, KNN and Random Forest
3. Tune the best model using recall of the disease class
4. Add a **safety layer** that uses predicted probabilities to flag high-risk and uncertain patients

##  Project Structure
```
heart-disease-capstone/
│── heart_disease.ipynb   # Full notebook (EDA, modeling, safety layer)
│── heart.csv             # Dataset
│── README.md             # Documentation
```

##  Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Pipeline, ColumnTransformer, GridSearchCV)
- Jupyter Notebook (VS Code)

##  Data Cleaning
- Original file: **1025 rows, 14 columns**, no missing values
- Found **723 duplicate rows** → only **302 unique rows** remained
- Duplicates were removed **before** the train/test split to prevent **data leakage** (otherwise the same patient appears in both train and test and accuracy looks falsely high)
- Target meaning in this dataset (verified from feature averages): **`0` = heart disease, `1` = no disease**

##  Workflow
1. Load data and run EDA (class balance, correlation heatmap)
2. Remove duplicates, then stratified 80/20 train-test split (241 / 61 rows)
3. Preprocessing with `ColumnTransformer`:
   - `StandardScaler` for continuous features (age, trestbps, chol, thalach, oldpeak)
   - `OneHotEncoder` for categorical features (sex, cp, fbs, restecg, exang, slope, ca, thal)
4. Compare models using 5-fold cross-validation
5. Tune with `GridSearchCV`, scoring on **recall of the disease class**
6. Choose the final model and add the safety layer

## Model Comparison (5-fold CV accuracy)
| Model | CV Accuracy |
|---|---|
| Logistic Regression | 0.834 |
| KNN | 0.838 |
| Random Forest | 0.842 |

The differences are smaller than the standard deviation (±0.03 to 0.04), so there is no clear winner on accuracy alone.

## Tuned Results (Test Set)
| Model | Test Accuracy | Disease cases missed (of 28) |
|---|---|---|
| **Logistic Regression** (C = 0.1) | **0.84** | **5** |
| Random Forest (tuned) | 0.80 | 7 |

**Final model: Logistic Regression**: simpler, more interpretable, and at least as good as Random Forest on this small dataset.

## Safety Layer
The model outputs the **probability of disease** (`predict_proba`). Instead of always giving a hard 0/1 answer:

| Disease probability | Decision |
|---|---|
| ≥ 0.60 | High risk, consult a doctor |
| ≤ 0.20 | Low risk |
| Between 0.20 and 0.60 | Uncertain, consult a doctor |

### Safety layer results (61 test patients)
| Decision | Total | Actually diseased | Actually healthy |
|---|---|---|---|
| High risk | 24 | 20 | 4 |
| Uncertain | 20 | 7 | 13 |
| Low risk | 17 | **1** | 16 |

- Without the safety layer: **5 of 28** diseased patients were missed
- With the safety layer: only **1 of 28** was labeled "Low risk", so **27 of 28 (~96%)** were flagged for a doctor
- Trade-off: more false alarms (17 healthy patients were also referred). In screening, this is an intentional choice.

### Example
```python
p = final_model.predict_proba(new_patient)[0, 0]   # probability of disease
print(safety_layer(p))
# Disease probability: 0.96 → High risk - consult a doctor
```

## Limitations
- Very small dataset (302 unique rows, only 28 diseased patients in the test set), so results can change a lot with a few cases
- Thresholds (0.60 / 0.20) were chosen by inspecting test results; a better approach is to choose them on training data using `cross_val_predict`
- Not validated on real clinical data, not for medical use

## Key Learnings
- Always check for duplicates before splitting data (data leakage)
- Never assume what `target = 1` means, verify it from the data
- Accuracy is not enough in medical problems; recall of the disease class matters most
- A simple model can beat a complex one on small data
- A safety layer (abstention) makes ML predictions safer to use

## How to Run
1. Clone the repository
   ```
   git clone https://github.com/Shazia645/heart-disease-risk-safety-layer.git
   ```
2. Install the required libraries
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open `heart_disease.ipynb` in VS Code or Jupyter Notebook and run all cells.

## Author
**Shazia Batool**
AI Student at COMSATS University | AI Intern at SATA Technologies
GitHub: [Shazia645](https://github.com/Shazia645)