import pandas as pd


df = pd.read_csv("merged_atp_matches_2000_2024.csv")


df_win = df.copy()

rename_cols = {
    "winner_name": "player_A_name",
    "loser_name": "player_B_name",
    "winner_rank": "player_A_rank",
    "loser_rank": "player_B_rank",
    "winner_rank_points": "player_A_rank_points",
    "loser_rank_points": "player_B_rank_points",
    "winner_hand": "player_A_hand",
    "loser_hand": "player_B_hand",
    "winner_age": "player_A_age",
    "loser_age": "player_B_age",
    "w_ace": "player_A_ace",
    "l_ace": "player_B_ace",
    "w_df": "player_A_df",
    "l_df": "player_B_df",
    "w_1stWon": "player_A_1stWon",
    "l_1stWon": "player_B_1stWon",
    "w_2ndWon": "player_A_2ndWon",
    "l_2ndWon": "player_B_2ndWon",
    "w_bpSaved": "player_A_bpSaved",
    "l_bpSaved": "player_B_bpSaved",
    "w_bpFaced": "player_A_bpFaced",
    "l_bpFaced": "player_B_bpFaced"
}
df_win.rename(columns=rename_cols, inplace=True)

df_win["target"] = 1

df_lose = df.copy()

rename_cols_loser = {
    "loser_name": "player_A_name",
    "winner_name": "player_B_name",
    "loser_rank": "player_A_rank",
    "winner_rank": "player_B_rank",
    "loser_rank_points": "player_A_rank_points",
    "winner_rank_points": "player_B_rank_points",
    "loser_hand": "player_A_hand",
    "winner_hand": "player_B_hand",
    "loser_age": "player_A_age",
    "winner_age": "player_B_age",
    "l_ace": "player_A_ace",
    "w_ace": "player_B_ace",
    "l_df": "player_A_df",
    "w_df": "player_B_df",
    "l_1stWon": "player_A_1stWon",
    "w_1stWon": "player_B_1stWon",
    "l_2ndWon": "player_A_2ndWon",
    "w_2ndWon": "player_B_2ndWon",
    "l_bpSaved": "player_A_bpSaved",
    "w_bpSaved": "player_B_bpSaved",
    "l_bpFaced": "player_A_bpFaced",
    "w_bpFaced": "player_B_bpFaced"
}
df_lose.rename(columns=rename_cols_loser, inplace=True)

df_lose["target"] = 0

df_final = pd.concat([df_win, df_lose], ignore_index=True)

df_final["rank_diff"] = df_final["player_A_rank"] - df_final["player_B_rank"]
df_final["rank_points_diff"] = df_final["player_A_rank_points"] - df_final["player_B_rank_points"]
df_final["age_diff"] = df_final["player_A_age"] - df_final["player_B_age"]
df_final["ace_diff"] = df_final["player_A_ace"] - df_final["player_B_ace"]
df_final["df_diff"] = df_final["player_A_df"] - df_final["player_B_df"]
df_final["first_serve_won_diff"] = df_final["player_A_1stWon"] - df_final["player_B_1stWon"]
df_final["second_serve_won_diff"] = df_final["player_A_2ndWon"] - df_final["player_B_2ndWon"]
df_final["bp_saved_diff"] = df_final["player_A_bpSaved"] - df_final["player_B_bpSaved"]
df_final["bp_faced_diff"] = df_final["player_A_bpFaced"] - df_final["player_B_bpFaced"]

df_final.to_csv("TRAINING_READY_DATASET.csv", index=False)

print("✅ Dataset finale salvato: TRAINING_READY_DATASET.csv")
print("Numero di righe:", len(df_final))
print("Numero di colonne:", len(df_final.columns))
print("Prime righe:")
print(df_final.head())
