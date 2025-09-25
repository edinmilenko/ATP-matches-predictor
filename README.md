# 🎾 Tennis Match Prediction – Australian Open 2025

This project applies **machine learning** techniques to predict the outcomes of men's tennis matches (ATP).  
The ultimate goal is to forecast the results of the **Australian Open 2025**, using historical ATP match data from 2000 to 2024.

---

## 📂 Project Structure

- **ATP_matches.csv** → main dataset containing ATP matches.
- **matches_input/** → folder with CSV files for the tournament draw (per round).
- **matches_output/** → folder where the predicted results are stored (per round).
- **main script** → data preprocessing, model training and predictions.

---

## ⚙️ Workflow

1. **Data Cleaning & Preparation**
   - Convert dates
   - Handle missing values
   - Encode categorical variables
   - Create new features (e.g. age difference, rank difference, ranking points difference, serve stats)

2. **Models Used**
   - **Random Forest** → baseline model, evaluated with accuracy, precision, recall, F1 and confusion matrix.
   - **XGBoost with BayesSearchCV** → final model with Bayesian hyperparameter tuning.

3. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC
   - Confusion Matrix
   - Feature importance (top 20 most influential features)

4. **Tournament Prediction**
   - Simulation of the Australian Open 2025 rounds (from 1st round → final).
   - Results saved as CSV with win probabilities and predicted winners.

---

## 📊 Expected Output

For each round, a CSV file is generated in the `matches_output/` folder with the following columns:

- `player_A_name`
- `player_B_name`
- `A_win_probability`
- `B_win_probability`
- `predicted_winner`

---

## 🛠️ Requirements

Install the required dependencies:

```bash
pip install pandas scikit-learn xgboost scikit-optimize category_encoders matplotlib seaborn
```

---

## ▶️ How to Run

1. Place the input files (`ATP_matches.csv` + `matches_input/` folder) in the project root.
2. Run the main script:
   ```bash
   python tennis_prediction.py
   ```
3. Results will be available in the `matches_output/` folder.

---
