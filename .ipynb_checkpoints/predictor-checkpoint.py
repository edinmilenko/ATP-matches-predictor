#!/usr/bin/env python
# coding: utf-8

# # Reading the CSV files and converting the columns in the correct format

import pandas as pd
#reading and showing off the dataset
matches = pd.read_csv("ATP_matches.csv")
#this is dataframe is going to be used later in the project
xg_matches = pd.read_csv("ATP_matches.csv")
matches.head(500000)

#Deleting rows that contain NaN values
matches = matches.dropna()
matches = matches.copy()  

#Converting date in datetime64ns type
matches["tourney_date"] = pd.to_datetime(
    matches["tourney_date"].astype(str),  
    format="mixed"                       
)

xg_matches["tourney_date"] = pd.to_datetime(
    xg_matches["tourney_date"].astype(str),  
    format="mixed"                      
)

#converting every object and float type into int type
matches.loc[:, "surface_code"] = matches["surface"].astype("category").cat.codes
matches.loc[:, "tourney_level_code"] = matches["tourney_level"].astype("category").cat.codes
matches.loc[:, "round_code"] = matches["round"].astype("category").cat.codes
matches.loc[:, "player_A_name_code"] = matches["player_A_name"].astype("category").cat.codes
matches.loc[:, "player_B_name_code"] = matches["player_B_name"].astype("category").cat.codes
matches.loc[:, "player_A_hand_code"] = matches["player_A_hand"].astype("category").cat.codes
matches.loc[:, "player_B_hand_code"] = matches["player_B_hand"].astype("category").cat.codes
matches["player_A_age"] = matches["player_A_age"].round().astype(int)
matches["player_B_age"] = matches["player_B_age"].round().astype(int)
matches["age_diff"] = matches["age_diff"].round().astype(int)
matches["player_A_rank"] = matches["player_A_rank"].round().astype(int)
matches["player_B_rank"] = matches["player_B_rank"].round().astype(int)
matches["rank_diff"] = matches["rank_diff"].round().astype(int)
matches["player_A_rank_points"] = matches["player_A_rank_points"].round().astype(int)
matches["player_B_rank_points"] = matches["player_B_rank_points"].round().astype(int)
matches["rank_points_diff"] = matches["rank_points_diff"].round().astype(int)

cols_to_round = [
    "player_A_ace", "player_A_df", "w_svpt", "w_1stIn", "player_A_1stWon",
    "player_A_2ndWon", "w_SvGms", "player_A_bpSaved", "player_A_bpFaced",
    "player_B_ace", "player_B_df", "l_svpt", "l_1stIn", "player_B_1stWon",
    "player_B_2ndWon", "l_SvGms", "player_B_bpSaved", "player_B_bpFaced",
    "player_A_rank.1", "player_A_rank_points.1", "player_B_rank.1",
    "player_B_rank_points.1", "rank_diff.1", "rank_points_diff.1",
    "age_diff.1", "ace_diff", "df_diff", "first_serve_won_diff",
    "second_serve_won_diff", "bp_saved_diff", "bp_faced_diff"
]

for col in cols_to_round:
    matches[col] = matches[col].round().astype(int)
pd.set_option('display.max_columns', None)
matches

# # Importing random forest, training the data and showing accuracy, precision, recall, F1-score and confusion matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

#n_estimators = the higher this is the more is takes to execute but potentially more accurate 
#min_samples_split = the higher this is the less accuracy but less overfitting
#random_state = if we run the random forest multiple time we are going to get the same result as long as the data remains the same
rf = RandomForestClassifier(n_estimators=500, min_samples_split=20, random_state=1)

train = matches[matches["tourney_date"] < '2022-01-01']

test = matches[matches["tourney_date"] > '2022-01-01']

predictors = ["draw_size", "player_A_age", "player_B_age", "age_diff", "player_A_rank", "player_B_rank", "rank_diff", "player_A_rank_points", "player_B_rank_points", "rank_points_diff", "surface_code", "tourney_level_code", "round_code", "player_A_name_code", "player_B_name_code", "player_A_hand_code", "player_B_hand_code"]

rf.fit(train[predictors], train["target"])

preds = rf.predict(test[predictors])

# accuracy
acc = accuracy_score(test["target"], preds)

acc

combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))

pd.crosstab(index=combined["actual"], columns=combined["prediction"])

# precision
precision_score(test["target"], preds)

# recall
recall_score(test["target"], preds)

# F1-score
f1_score(test["target"], preds)

# Confusion Matrix
cm = confusion_matrix(test["target"], preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# # Trying to improve the parameters by using rolling averages

#showing an example with player "Novak Djokovic"
grouped_matches = matches.groupby("player_A_name")
group = grouped_matches.get_group("Novak Djokovic")

group

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("tourney_date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ['player_A_ace', 'player_A_df', 'w_svpt', 'w_1stIn', 'player_A_1stWon', 
    'player_A_2ndWon', 'w_SvGms', 'player_A_bpSaved', 'player_A_bpFaced', 
    'player_B_ace', 'player_B_df', 'l_svpt', 'l_1stIn', 'player_B_1stWon', 
    'player_B_2ndWon', 'l_SvGms', 'player_B_bpSaved', 'player_B_bpFaced', 
    'player_A_rank.1', 'player_A_rank_points.1', 'player_B_rank.1', 
    'player_B_rank_points.1', 'rank_diff.1', 'rank_points_diff.1', 
    'age_diff.1', 'ace_diff', 'df_diff', 'first_serve_won_diff', 
    'second_serve_won_diff', 'bp_saved_diff', 'bp_faced_diff']
new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)

matches_rolling = matches.groupby("player_A_name").apply(lambda x: rolling_averages(x, cols, new_cols))

matches_rolling

#Fixing indexes
matches_rolling = matches_rolling.reset_index(drop=True)

matches_rolling

def make_predictions(data, predictiors):
    train = data[data["tourney_date"] < '2023-01-01']
    test = data[data["tourney_date"] > '2023-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision_roll_avg = precision_score(test["target"], preds)
    acc_roll_avg = accuracy_score(test["target"], preds)
    recall_roll_avg = recall_score(test["target"], preds)
    f1_roll_avg = f1_score(test["target"], preds)
    cm_roll_avg = confusion_matrix(test["target"], preds)
    disp_roll_avg = ConfusionMatrixDisplay(confusion_matrix=cm_roll_avg)
    return combined, precision_roll_avg, acc_roll_avg, recall_roll_avg, f1_roll_avg, disp_roll_avg

combined, precision_roll_avg, acc_roll_avg, recall_roll_avg, f1_roll_avg, disp_roll_avg = make_predictions(matches_rolling, predictors + new_cols)

#slightly decreased
acc_roll_avg

#slightly improved
precision_roll_avg

#slightly improved
recall_roll_avg

#slightly improved
f1_roll_avg

disp_roll_avg.plot()

#showing the results in a summary table
combined = combined.merge(matches_rolling[["tourney_date", "player_A_name", "player_B_name", "target"]], left_index=True, right_index=True)

combined


#Overall rolling averages slightly improved our model
#Now XGBoost algorithm will be tested on the initial dataset to see if it is eligible to predict future matches

# # Preparing the data for training and creating train and test sets

# In[47]:


#we will be using xgboost on the dataset "xg_matches" created at the beginning of the project 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 


X = xg_matches.drop(columns='target') 
y = xg_matches['target'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=8) 

#when the error on opt() shows up, just de-comment this cell and run it once and it converts the time in the EPOC format and drops the old tourney_date column, then comment it again and run again this cell 
#xg_matches["tourney_date_epoch"] = xg_matches["tourney_date"].astype("int64") // 10**9 
#xg_matches = xg_matches.drop(columns=["tourney_date"])


# # Building a pipeline of training

from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBClassifier

estimators = [
    ('encoder', TargetEncoder()), 
    ('clf', XGBClassifier(random_state=8)) #can customize objective function with the objective parameter
]
pipe = Pipeline(steps=estimators)
pipe


# # Setting up hyperparameter tuning
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

search_space = {
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode' : Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0)
}

#cv is the number of folds in cross validation, n_iter is the number of hyperparameters settings that are sampled as 10, scoring is the metric of evaluation
#Bayesian search is a method that leverages Bayesian statistics to efficiently find a target in a large, uncertain space by combining prior knowledge with new evidence to continuously update the probability of the target's location
opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=8) 

# Train the XGBoost model
# In[50]:


opt.fit(X_train, y_train)


# # Evaluating the model and making predictions

opt.best_estimator_

opt.best_score_

opt.score(X_test, y_test)


y_pred = opt.predict(X_test)
y_pred


opt.predict_proba(X_test)

#measure feature importance
opt.best_estimator_.steps

acc_xg = accuracy_score(y_test, y_pred)
acc_xg

precision_xg = precision_score(y_test, y_pred, average='weighted')
precision_xg

recall_xg = recall_score(y_test, y_pred, average='weighted')
recall_xg

f1_xg = f1_score(y_test, y_pred, average='weighted')
f1_xg

cm_xg = confusion_matrix(y_test, y_pred)
disp_xg = ConfusionMatrixDisplay(confusion_matrix=cm_xg)
disp_xg.plot()

# # Measuring feature importance

#showing only the top 20 most used features so the graph is readable
from xgboost import plot_importance
import matplotlib.pyplot as plt

xgboost_step = opt.best_estimator_.steps[1]
xgboost_model = xgboost_step[1]

plt.figure(figsize=(10, 8))  # ingrandisce la figura
plot_importance(xgboost_model, max_num_features=20)  # mostra solo le 20 più importanti
plt.show()

import seaborn as sns

# Checking whether the dataset is balanced or not to see if the metrics calculated before are reliable
sns.countplot(x=y)
plt.title("Distribution of classes in the dataset")
plt.xlabel("Target class")
plt.ylabel("Number of samples")
plt.show()


# # Now, using xgboost algorithm, we are gonna try to predict the australian open 2025 using data up to december 2024

column_to_keep = [
    'surface', 'tourney_level', 'round', 'draw_size', 'tourney_date_epoch',
    'player_A_name', 'player_B_name', 'player_A_hand', 'player_B_hand',
    'player_A_age', 'player_B_age', 'age_diff', 'player_A_rank', 'player_B_rank',
    'rank_diff', 'player_A_rank_points', 'player_B_rank_points',
    'rank_points_diff', 'target'
]

first_round = pd.read_csv("matches_input/first_round_matches.csv")
train_data = xg_matches[column_to_keep].copy()

X_train_2024 = train_data.drop(columns=["target"])
y_train_2024 = train_data["target"]

opt.fit(X_train_2024, y_train_2024)

results_first_rounds = first_round.copy()

#predictions
predictions = opt.predict_proba(first_round)
winner_predictions = opt.predict(first_round)

results_first_rounds['A_win_probability'] = predictions[:, 1]  # probability player A wins
results_first_rounds['B_win_probability'] = predictions[:, 0]  # probability player B wins
results_first_rounds['predicted_winner'] = winner_predictions

results_first_rounds


#Saving matches results in a CSV file
colonne_da_scaricare = ['player_A_name', 'player_B_name', 'A_win_probability', 'B_win_probability', 'predicted_winner']
res = results_F_rounds[colonne_da_scaricare]
res.to_csv('first_round_results.csv', index=False)

second_round = pd.read_csv("matches_input/second_round_matches.csv")

results_second_rounds = second_round.copy()

predictions = opt.predict_proba(second_round)
winner_predictions = opt.predict(second_round)

results_second_rounds['A_win_probability'] = predictions[:, 1]  
results_second_rounds['B_win_probability'] = predictions[:, 0]  
results_second_rounds['predicted_winner'] = winner_predictions

pd.set_option('display.max_rows', None)

results_second_rounds

colonne_da_scaricare = ['player_A_name', 'player_B_name', 'A_win_probability', 'B_win_probability', 'predicted_winner']
res = results_F_rounds[colonne_da_scaricare]
res.to_csv('second_round_results.csv', index=False)

third_round = pd.read_csv("matches_input/third_round_matches.csv")

results_third_rounds = third_round.copy()

predictions = opt.predict_proba(third_round)
winner_predictions = opt.predict(third_round)

results_third_rounds['A_win_probability'] = predictions[:, 1]  
results_third_rounds['B_win_probability'] = predictions[:, 0] 
results_third_rounds['predicted_winner'] = winner_predictions

pd.set_option('display.max_rows', None)

results_third_rounds

colonne_da_scaricare = ['player_A_name', 'player_B_name', 'A_win_probability', 'B_win_probability', 'predicted_winner']
res = results_F_rounds[colonne_da_scaricare]
res.to_csv('third_round_results.csv', index=False)

fourth_round = pd.read_csv("matches_input/fourth_round_matches.csv")

results_fourth_rounds = fourth_round.copy()

predictions = opt.predict_proba(fourth_round)
winner_predictions = opt.predict(fourth_round)

results_fourth_rounds['A_win_probability'] = predictions[:, 1]  
results_fourth_rounds['B_win_probability'] = predictions[:, 0]  
results_fourth_rounds['predicted_winner'] = winner_predictions

pd.set_option('display.max_rows', None)

results_fourth_rounds

colonne_da_scaricare = ['player_A_name', 'player_B_name', 'A_win_probability', 'B_win_probability', 'predicted_winner']
res = results_F_rounds[colonne_da_scaricare]
res.to_csv('fourth_round_results.csv', index=False)

QF_round = pd.read_csv("matches_input/QF_matches.csv")

results_QF_rounds = QF_round.copy()

predictions = opt.predict_proba(QF_round)
winner_predictions = opt.predict(QF_round)

results_QF_rounds['A_win_probability'] = predictions[:, 1]  
results_QF_rounds['B_win_probability'] = predictions[:, 0]  
results_QF_rounds['predicted_winner'] = winner_predictions

pd.set_option('display.max_rows', None)

results_QF_rounds

colonne_da_scaricare = ['player_A_name', 'player_B_name', 'A_win_probability', 'B_win_probability', 'predicted_winner']
res = results_F_rounds[colonne_da_scaricare]
res.to_csv('QF_results.csv', index=False)

SF_round = pd.read_csv("matches_input/SF_matches.csv")

results_SF_rounds = SF_round.copy()

predictions = opt.predict_proba(SF_round)
winner_predictions = opt.predict(SF_round)

results_SF_rounds['A_win_probability'] = predictions[:, 1]  
results_SF_rounds['B_win_probability'] = predictions[:, 0] 
results_SF_rounds['predicted_winner'] = winner_predictions

pd.set_option('display.max_rows', None)

results_SF_rounds


# In[103]:


colonne_da_scaricare = ['player_A_name', 'player_B_name', 'A_win_probability', 'B_win_probability', 'predicted_winner']
res = results_F_rounds[colonne_da_scaricare]

res.to_csv('SF_results.csv', index=False)

F_round = pd.read_csv("matches_input/F_matches.csv")

results_F_rounds = F_round.copy()

predictions = opt.predict_proba(F_round)
winner_predictions = opt.predict(F_round)

results_F_rounds['A_win_probability'] = predictions[:, 1]  
results_F_rounds['B_win_probability'] = predictions[:, 0] 
results_F_rounds['predicted_winner'] = winner_predictions

pd.set_option('display.max_rows', None)

results_F_rounds

colonne_da_scaricare = ['player_A_name', 'player_B_name', 'A_win_probability', 'B_win_probability', 'predicted_winner']
res = results_F_rounds[colonne_da_scaricare]

res.to_csv('F_results.csv', index=False)