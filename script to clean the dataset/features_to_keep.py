import pandas as pd

matches = pd.read_csv("TRAINING_READY_DATASET.csv")

columns_to_keep = [

    "surface", "tourney_level", "round", "draw_size", "tourney_date",

    "player_A_name", "player_B_name",
    "player_A_hand", "player_B_hand",
    "player_A_age", "player_B_age", "age_diff",

    "player_A_rank", "player_B_rank", "rank_diff",
    "player_A_rank_points", "player_B_rank_points", "rank_points_diff",

    "player_A_ace",
    "player_A_df",
    "w_svpt",
    "w_1stIn",
    "player_A_1stWon",
    "player_A_2ndWon",
    "w_SvGms",
    "player_A_bpSaved",
    "player_A_bpFaced",
    "player_B_ace",
    "player_B_df",
    "l_svpt",
    "l_1stIn",
    "player_B_1stWon",
    "player_B_2ndWon",
    "l_SvGms",
    "player_B_bpSaved",
    "player_B_bpFaced",
    "player_A_rank",
    "player_A_rank_points",
    "player_B_rank",
    "player_B_rank_points",
    "rank_diff",
    "rank_points_diff",
    "age_diff",
    "ace_diff",
    "df_diff",
    "first_serve_won_diff",
    "second_serve_won_diff",
    "bp_saved_diff",
    "bp_faced_diff",
    # Target
    "target"
]

matches_clean = matches[columns_to_keep].copy()

matches_clean.to_csv("ONLY_IMPORTANT_FEATURES.csv", index=False)

print(f"✅ Dataset pulito salvato! Dimensioni finali: {matches_clean.shape}")
