# ATP-matches-predictor



\# 🎾 Tennis Match Prediction – Australian Open 2025



This project applies machine learning techniques to predict the outcomes of men's tennis matches (ATP).  

The ultimate goal is to forecast the results of the Australian Open 2025, using historical ATP match data from 2000 to 2024.



---



\## 📂 Project Structure



\- \*\*ATP\_matches.csv\*\* → main dataset containing ATP matches.

\- \*\*matches\_input/\*\* → folder with CSV files for the tournament draw (per round).

\- \*\*matches\_output/\*\* → folder where the predicted results are stored (per round).

\- \*\*main script\*\* → data preprocessing, model training and predictions.



---



\## ⚙️ Workflow



1\. \*\*Data Cleaning \& Preparation\*\*

&nbsp;  - Convert dates

&nbsp;  - Handle missing values

&nbsp;  - Encode categorical variables

&nbsp;  - Create new features (e.g. age difference, rank difference, ranking points difference, serve stats)



2\. \*\*Models Used\*\*

&nbsp;  - \*\*Random Forest\*\* → baseline model, evaluated with accuracy, precision, recall, F1 and confusion matrix.

&nbsp;  - \*\*XGBoost with BayesSearchCV\*\* → final model with Bayesian hyperparameter tuning.



3\. \*\*Evaluation\*\*

&nbsp;  - Accuracy, Precision, Recall, F1-score

&nbsp;  - ROC-AUC

&nbsp;  - Confusion Matrix

&nbsp;  - Feature importance (top 20 most influential features)



4\. \*\*Tournament Prediction\*\*

&nbsp;  - Simulation of the Australian Open 2025 rounds (from 1st round → final).

&nbsp;  - Results saved as CSV with win probabilities and predicted winners.



---



\## 📊 Expected Output



For each round, a CSV file is generated in the `matches\_output/` folder with the following columns:



\- `player\_A\_name`

\- `player\_B\_name`

\- `A\_win\_probability`

\- `B\_win\_probability`

\- `predicted\_winner`



---



\## 🛠️ Requirements



Install the required dependencies:



```bash

pip install pandas scikit-learn xgboost scikit-optimize category\_encoders matplotlib seaborn

```



---



\## ▶️ How to Run



1\. Place the input files (`ATP\_matches.csv` + `matches\_input/` folder) in the project root.

2\. Run the main script:

&nbsp;  ```bash

&nbsp;  python tennis\_prediction.py

&nbsp;  ```

3\. Results will be available in the `matches\_output/` folder.



---

