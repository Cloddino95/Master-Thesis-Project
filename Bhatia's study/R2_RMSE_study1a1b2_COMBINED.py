# from combined_risk_rating_1A import results_df_A_combo
# from combined_risk_rating_1B import results_df_B_combo
# from combined_risk_rating_2 import results_df_2_combo
import pandas as pd

df1a_combo = pd.read_csv('/CSV_results/results_df_1A_COMBINED.csv')
df1b_combo = pd.read_csv('/CSV_results/results_df_1B_COMBINED.csv')
df2_combo = pd.read_csv('/CSV_results/results_df_2_COMBINED.csv')

df1a_combo = df1a_combo.drop("parameter", axis=1)
df1a_combo_mean = df1a_combo.groupby('model').mean().reset_index()
df1a_combo_mean.insert(0, "Study", "Study 1A")

df1b_combo = df1b_combo.drop("parameter", axis=1)
df1b_combo_mean = df1b_combo.groupby('model').mean().reset_index()
df1b_combo_mean.insert(0, "Study", "Study 1B")

df2_combo = df2_combo.drop("parameter", axis=1)
df2_combo_mean = df2_combo.groupby('model').mean().reset_index()
df2_combo_mean.insert(0, "Study", "Study 2")

# all_result_df_combo = pd.concat([results_df_A_combo, results_df_B_combo, results_df_2_combo], axis=0)
all_result_df_combo_csv = pd.concat([df1a_combo_mean, df1b_combo_mean, df2_combo_mean], axis=0)

