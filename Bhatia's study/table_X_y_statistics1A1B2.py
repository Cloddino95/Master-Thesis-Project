import pandas as pd

# --------------------SUMMARY STATISTICS FOR STUDIES 1A, 1B, 2 (X) AND (y)--------------------
from risk_rating_1A_customized_K_C_AGG import mean_statistics_1A_X, statistics_1A_y
from risk_rating_1B_customized_K_C_AGG import mean_statistics_1B_X, statistics_1B_y
from risk_rating_2_customized_K_C_AGG import mean_statistics_2_X, statistics_2_y
from psychometric_1A import mean_psyco_1A_X
from psychometric_1B import mean_psyco_1B_X
from psychometric_2 import mean_psyco_2_X

sum_statistics = pd.concat([mean_statistics_1A_X, mean_statistics_1B_X, mean_statistics_2_X, mean_psyco_1A_X, mean_psyco_1B_X,
                            mean_psyco_2_X, statistics_1A_y, statistics_1B_y, statistics_2_y], axis=1)
col_names = ['study 1A (X)', 'study 1B (X)', 'study 2 (X)', 'psycho 1A (X)', 'psycho 1B (X)', 'psycho 2 (X)', 'study 1A (y)',
             'study 1B (y)', 'study 2 (y)']
sum_statistics.columns = col_names
T_sum_statistics = sum_statistics.T
T_sum_statistics.drop(columns=['25%', '75%'], inplace=True)
T_sum_statistics = T_sum_statistics.rename(columns={'50%': 'median'})
T_sum_statistics.to_csv('Summary_statistics_Study1A1B2psy.csv')
