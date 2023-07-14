from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import gensim
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score  # , RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import shapiro
from combined_risk_rating_1A import Xa, ya

model = gensim.models.KeyedVectors.load(
    '/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

X_main = Xa
y_main = ya

X1, X_temp, y1, y_temp = train_test_split(X_main, y_main, test_size=0.66, random_state=42)
X2, X3, y2, y3 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

parameter_values = {
    'SVR-RBF': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'SVR-polynomial': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'SVR-sigmoid': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Lasso Regression': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'Ridge Regression': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'KNN Regression': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

models = {'SVR-RBF': SVR(kernel='rbf'),
          'SVR-polynomial': SVR(kernel='poly'),
          'SVR-sigmoid': SVR(kernel='sigmoid'),
          'Lasso Regression': Lasso(),
          'Ridge Regression': Ridge(),
          'KNN Regression': KNeighborsRegressor()}

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

results_list_2 = []
datasets_X = [X1, X2, X3]
datasets_y = [y1, y2, y3]
subset_names = ['Subset1', 'Subset2', 'Subset3']

for subset_name, (X, y) in zip(subset_names, zip(datasets_X, datasets_y)):
    for model_name, model in tqdm(models.items()):
        if model_name == 'KNN Regression':
            for k in parameter_values[model_name]:
                model.set_params(n_neighbors=k)
                r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
                mse_scores = - cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
                results_list_2.append({'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(r2_scores),
                                       'std_score_R2': np.std(r2_scores), 'mean_score_MSE': np.mean(mse_scores),
                                       'std_score_MSE': np.std(mse_scores), 'subset': subset_name})
        else:
            for C in tqdm(parameter_values[model_name]):
                model.set_params(alpha=C) if model_name in ['Lasso Regression',
                                                            'Ridge Regression'] else model.set_params(C=C)
                r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
                mse_scores = - cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
                results_list_2.append(
                    {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                     'std_score_R2': np.std(r2_scores), 'mean_score_MSE': np.mean(np.sqrt(mse_scores)),
                     'std_score_MSE': np.std(np.sqrt(mse_scores)), 'subset': subset_name})

results_df_2_Bhatia = pd.DataFrame(results_list_2, index=range(len(results_list_2)))
# results_df_2_Bhatia.to_csv('results_df_2_Bhatia.csv', index=False) # Uncomment to save the results to csv file

df = results_df_2_Bhatia
# Group by 'model' and 'subset' and calculate the mean of the four columns
df_means_results = df.groupby(['model', 'subset']).mean()[
    ['mean_score_R2', 'std_score_R2', 'mean_score_MSE', 'std_score_MSE']]
# Reset index to make 'model' and 'subset' normal columns again
df_means_results = df_means_results.reset_index()

# ------------------------ PERFORMING THE SAME SPLIT (3 DATASETS) FOR THE PSYCHOMETRIC APPR0ACH ------------------------

from psychometric_2 import X, y

X_psy = X
y_psy = y

X1_psy, X_temp_psy, y1_psy, y_temp_psy = train_test_split(X_psy, y_psy, test_size=0.66, random_state=42)
X2_psy, X3_psy, y2_psy, y3_psy = train_test_split(X_temp_psy, y_temp_psy, test_size=0.5, random_state=42)

datasets_X_psy = [X1_psy, X2_psy, X3_psy]
datasets_y_psy = [y1_psy, y2_psy, y3_psy]

model_psy = LinearRegression()
results_list_2_psy = []

for subset_name, (X_psy, y_psy) in zip(subset_names, zip(datasets_X_psy, datasets_y_psy)):
    model_psy.fit(X_psy, y_psy)
    y_pred = model_psy.predict(X_psy)
    r2_scores = r2_score(y_psy, y_pred)
    mse_scores = mean_squared_error(y_psy, y_pred)
    results_list_2_psy.append({'model': 'Psychometric Approach', 'mean_score_R2': r2_scores,
                               'mean_score_MSE': np.sqrt(mse_scores), 'subset': subset_name})

results_df_psy2 = pd.DataFrame(results_list_2_psy, range(len(results_list_2_psy)))
results_df_psy2.set_index('subset', inplace=True)
# -------------------------------- COMPARING THE RESULTS OF THE TWO APPROACHES & T TEST -------------------------------

# Dividing the df_means_results per each model
df_means_SVR_RBF = df_means_results[df_means_results['model'] == 'SVR-RBF']
df_means_SVR_RBF.set_index('subset', inplace=True)  # Setting the index to 'subset' to be able to make operation between the two pd.dataframes
df_means_SVR_polynomial = df_means_results[df_means_results['model'] == 'SVR-polynomial']
df_means_SVR_polynomial.set_index('subset', inplace=True)
df_means_SVR_sigmoid = df_means_results[df_means_results['model'] == 'SVR-sigmoid']
df_means_SVR_sigmoid.set_index('subset', inplace=True)
df_means_Lasso_Regression = df_means_results[df_means_results['model'] == 'Lasso Regression']
df_means_Lasso_Regression.set_index('subset', inplace=True)
df_means_Ridge_Regression = df_means_results[df_means_results['model'] == 'Ridge Regression']
df_means_Ridge_Regression.set_index('subset', inplace=True)
df_means_KNN_Regression = df_means_results[df_means_results['model'] == 'KNN Regression']
df_means_KNN_Regression.set_index('subset', inplace=True)

# make the difference of mean_score_R2 and mean_score_MSE between the two approaches
diffR2_RBF_psy = df_means_SVR_RBF['mean_score_R2'] - results_df_psy2['mean_score_R2']
diffMSE_RBF_psy = df_means_SVR_RBF['mean_score_MSE'] - results_df_psy2['mean_score_MSE']
diffR2_polynomial_psy = df_means_SVR_polynomial['mean_score_R2'] - results_df_psy2['mean_score_R2']
diffMSE_polynomial_psy = df_means_SVR_polynomial['mean_score_MSE'] - results_df_psy2['mean_score_MSE']
diffR2_sigmoid_psy = df_means_SVR_sigmoid['mean_score_R2'] - results_df_psy2['mean_score_R2']
diffMSE_sigmoid_psy = df_means_SVR_sigmoid['mean_score_MSE'] - results_df_psy2['mean_score_MSE']
diffR2_Lasso_psy = df_means_Lasso_Regression['mean_score_R2'] - results_df_psy2['mean_score_R2']
diffMSE_Lasso_psy = df_means_Lasso_Regression['mean_score_MSE'] - results_df_psy2['mean_score_MSE']
diffR2_Ridge_psy = df_means_Ridge_Regression['mean_score_R2'] - results_df_psy2['mean_score_R2']
diffMSE_Ridge_psy = df_means_Ridge_Regression['mean_score_MSE'] - results_df_psy2['mean_score_MSE']
diffR2_KNN_psy = df_means_KNN_Regression['mean_score_R2'] - results_df_psy2['mean_score_R2']
diffMSE_KNN_psy = df_means_KNN_Regression['mean_score_MSE'] - results_df_psy2['mean_score_MSE']

# ---------------------------------------- Performing Shapiro-Wilk test for normality ---------------------------------
"""
in general, it is possible to perform paired t-test if the differences are normally distributed. 
Shapiro-Wilk test: This is a common statistical test for normality. The null hypothesis for this test is that the data is 
normally distributed. So if the p-value from the test is less than the significance level (often 0.05), then you reject 
the null hypothesis and conclude that the data is not normally distributed.
"""

stat, p = shapiro(diffR2_RBF_psy)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('R2_RBF_psy: Sample looks Gaussian (fail to reject H0)')
else:
    print('R2_RBF_psy: Sample does not look Gaussian (reject H0)')

diffR2_RBF_psy_stat, diffR2_RBF_psy_p = shapiro(diffR2_RBF_psy)
diffMSE_RBF_psy_stat, diffMSE_RBF_psy_p = shapiro(diffMSE_RBF_psy)
diffR2_polynomial_psy_stat, diffR2_polynomial_psy_p = shapiro(diffR2_polynomial_psy)
diffMSE_polynomial_psy_stat, diffMSE_polynomial_psy_p = shapiro(diffMSE_polynomial_psy)
diffR2_sigmoid_psy_stat, diffR2_sigmoid_psy_p = shapiro(diffR2_sigmoid_psy)
diffMSE_sigmoid_psy_stat, diffMSE_sigmoid_psy_p = shapiro(diffMSE_sigmoid_psy)
diffR2_Lasso_psy_stat, diffR2_Lasso_psy_p = shapiro(diffR2_Lasso_psy)
diffMSE_Lasso_psy_stat, diffMSE_Lasso_psy_p = shapiro(diffMSE_Lasso_psy)
diffR2_Ridge_psy_stat, diffR2_Ridge_psy_p = shapiro(diffR2_Ridge_psy)
diffMSE_Ridge_psy_stat, diffMSE_Ridge_psy_p = shapiro(diffMSE_Ridge_psy)
diffR2_KNN_psy_stat, diffR2_KNN_psy_p = shapiro(diffR2_KNN_psy)
diffMSE_KNN_psy_stat, diffMSE_KNN_psy_p = shapiro(diffMSE_KNN_psy)

# save the results of the Shapiro-Wilk test in a dataframe
shapiro_test = pd.DataFrame(
    {'model': ['SVR-RBF', 'SVR-polynomial', 'SVR-sigmoid', 'Lasso Regression', 'Ridge Regression', 'KNN Regression'],
     'statistic_R2': [diffR2_RBF_psy_stat, diffR2_polynomial_psy_stat, diffR2_sigmoid_psy_stat, diffR2_Lasso_psy_stat,
                      diffR2_Ridge_psy_stat, diffR2_KNN_psy_stat],
     'p_value_R2': [diffR2_RBF_psy_p, diffR2_polynomial_psy_p, diffR2_sigmoid_psy_p, diffR2_Lasso_psy_p,
                    diffR2_Ridge_psy_p, diffR2_KNN_psy_p],
     'statistic_MSE': [diffMSE_RBF_psy_stat, diffMSE_polynomial_psy_stat, diffMSE_sigmoid_psy_stat,
                       diffMSE_Lasso_psy_stat,
                       diffMSE_Ridge_psy_stat, diffMSE_KNN_psy_stat],
     'p_value_MSE': [diffMSE_RBF_psy_p, diffMSE_polynomial_psy_p, diffMSE_sigmoid_psy_p, diffMSE_Lasso_psy_p,
                     diffMSE_Ridge_psy_p, diffMSE_KNN_psy_p]})
shapiro_test.to_csv('shapiro_test_COMBINED.csv')

# ---------------------------------------- Performing one-sample & paired t-test ---------------------------------------

# Performing one-sample t-test

RBF_psy_t1_statistic_r2, RBF_psy_p1_value_r2 = stats.ttest_1samp(diffR2_RBF_psy, 0)
RBF_psy_t1_statistic_mse, RBF_psy_p1_value_mse = stats.ttest_1samp(diffMSE_RBF_psy, 0)
polynomial_psy_t1_statistic_r2, polynomial_psy_p1_value_r2 = stats.ttest_1samp(diffR2_polynomial_psy, 0)
polynomial_psy_t1_statistic_mse, polynomial_psy_p1_value_mse = stats.ttest_1samp(diffMSE_polynomial_psy, 0)
sigmoid_psy_t1_statistic_r2, sigmoid_psy_p1_value_r2 = stats.ttest_1samp(diffR2_sigmoid_psy, 0)
sigmoid_psy_t1_statistic_mse, sigmoid_psy_p1_value_mse = stats.ttest_1samp(diffMSE_sigmoid_psy, 0)
Lasso_psy_t1_statistic_r2, Lasso_psy_p1_value_r2 = stats.ttest_1samp(diffR2_Lasso_psy, 0)
Lasso_psy_t1_statistic_mse, Lasso_psy_p1_value_mse = stats.ttest_1samp(diffMSE_Lasso_psy, 0)
Ridge_psy_t1_statistic_r2, Ridge_psy_p1_value_r2 = stats.ttest_1samp(diffR2_Ridge_psy, 0)
Ridge_psy_t1_statistic_mse, Ridge_psy_p1_value_mse = stats.ttest_1samp(diffMSE_Ridge_psy, 0)
KNN_psy_t1_statistic_r2, KNN_psy_p1_value_r2 = stats.ttest_1samp(diffR2_KNN_psy, 0)
KNN_psy_t1_statistic_mse, KNN_psy_p1_value_mse = stats.ttest_1samp(diffMSE_KNN_psy, 0)

# saving the one-sample t-test results in a dataframe
one_sample_ttest = pd.DataFrame(
    {'model': ['SVR-RBF', 'SVR-polynomial', 'SVR-sigmoid', 'Lasso Regression', 'Ridge Regression', 'KNN Regression'],
     't_statistic_R2': [RBF_psy_t1_statistic_r2, polynomial_psy_t1_statistic_r2, sigmoid_psy_t1_statistic_r2,
                        Lasso_psy_t1_statistic_r2, Ridge_psy_t1_statistic_r2, KNN_psy_t1_statistic_r2],
     'p_value_R2': [RBF_psy_p1_value_r2, polynomial_psy_p1_value_r2, sigmoid_psy_p1_value_r2, Lasso_psy_p1_value_r2,
                    Ridge_psy_p1_value_r2, KNN_psy_p1_value_r2],
     't_statistic_MSE': [RBF_psy_t1_statistic_mse, polynomial_psy_t1_statistic_mse, sigmoid_psy_t1_statistic_mse,
                         Lasso_psy_t1_statistic_mse, Ridge_psy_t1_statistic_mse, KNN_psy_t1_statistic_mse],
     'p_value_MSE': [RBF_psy_p1_value_mse, polynomial_psy_p1_value_mse, sigmoid_psy_p1_value_mse,
                     Lasso_psy_p1_value_mse, Ridge_psy_p1_value_mse, KNN_psy_p1_value_mse]})
# ttest_psy1.to_csv('ttest_psy1.csv')

# Performing paired sample t-tests
RBF_psy_t2_statistic_r2, RBF_psy_p2_value_r2 = stats.ttest_rel(df_means_SVR_RBF['mean_score_R2'],
                                                               results_df_psy2['mean_score_R2'])
RBF_psy_t2_statistic_mse, RBF_psy_p2_value_mse = stats.ttest_rel(df_means_SVR_RBF['mean_score_MSE'],
                                                                 results_df_psy2['mean_score_MSE'])
polynomial_psy_t2_statistic_r2, polynomial_psy_p2_value_r2 = stats.ttest_rel(df_means_SVR_polynomial['mean_score_R2'],
                                                                             results_df_psy2['mean_score_R2'])
polynomial_psy_t2_statistic_mse, polynomial_psy_p2_value_mse = stats.ttest_rel(
    df_means_SVR_polynomial['mean_score_MSE'],
    results_df_psy2['mean_score_MSE'])
sigmoid_psy_t2_statistic_r2, sigmoid_psy_p2_value_r2 = stats.ttest_rel(df_means_SVR_sigmoid['mean_score_R2'],
                                                                       results_df_psy2['mean_score_R2'])
sigmoid_psy_t2_statistic_mse, sigmoid_psy_p2_value_mse = stats.ttest_rel(df_means_SVR_sigmoid['mean_score_MSE'],
                                                                         results_df_psy2['mean_score_MSE'])
Lasso_psy_t2_statistic_r2, Lasso_psy_p2_value_r2 = stats.ttest_rel(df_means_Lasso_Regression['mean_score_R2'],
                                                                   results_df_psy2['mean_score_R2'])
Lasso_psy_t2_statistic_mse, Lasso_psy_p2_value_mse = stats.ttest_rel(df_means_Lasso_Regression['mean_score_MSE'],
                                                                     results_df_psy2['mean_score_MSE'])
Ridge_psy_t2_statistic_r2, Ridge_psy_p2_value_r2 = stats.ttest_rel(df_means_Ridge_Regression['mean_score_R2'],
                                                                   results_df_psy2['mean_score_R2'])
Ridge_psy_t2_statistic_mse, Ridge_psy_p2_value_mse = stats.ttest_rel(df_means_Ridge_Regression['mean_score_MSE'],
                                                                     results_df_psy2['mean_score_MSE'])
KNN_psy_t2_statistic_r2, KNN_psy_p2_value_r2 = stats.ttest_rel(df_means_KNN_Regression['mean_score_R2'],
                                                               results_df_psy2['mean_score_R2'])
KNN_psy_t2_statistic_mse, KNN_psy_p2_value_mse = stats.ttest_rel(df_means_KNN_Regression['mean_score_MSE'],
                                                                 results_df_psy2['mean_score_MSE'])

# saving the paired sample t-test results in a dataframe
paired_sample_ttest_comb_1a = pd.DataFrame(
    {'model': ['SVR-RBF', 'SVR-polynomial', 'SVR-sigmoid', 'Lasso Regression', 'Ridge Regression', 'KNN Regression'],
     't_statistic_R2': [RBF_psy_t2_statistic_r2, polynomial_psy_t2_statistic_r2, sigmoid_psy_t2_statistic_r2,
                        Lasso_psy_t2_statistic_r2, Ridge_psy_t2_statistic_r2, KNN_psy_t2_statistic_r2],
     'p_value_R2': [RBF_psy_p2_value_r2, polynomial_psy_p2_value_r2, sigmoid_psy_p2_value_r2, Lasso_psy_p2_value_r2,
                    Ridge_psy_p2_value_r2, KNN_psy_p2_value_r2],
     't_statistic_MSE': [RBF_psy_t2_statistic_mse, polynomial_psy_t2_statistic_mse, sigmoid_psy_t2_statistic_mse,
                         Lasso_psy_t2_statistic_mse, Ridge_psy_t2_statistic_mse, KNN_psy_t2_statistic_mse],
     'p_value_MSE': [RBF_psy_p2_value_mse, polynomial_psy_p2_value_mse, sigmoid_psy_p2_value_mse,
                     Lasso_psy_p2_value_mse, Ridge_psy_p2_value_mse, KNN_psy_p2_value_mse]})
paired_sample_ttest_comb_1a.to_csv('paired_sample_Ttest_NOcorrection_COMBINED_1A.csv')

# applying Bonferroni correction on the above paired sample t-test results df

Bonfcorrect_Ttest_1a_comb = paired_sample_ttest_comb_1a.copy()

# columns to be multiplied
cols_to_multiply = ['p_value_R2', 'p_value_MSE']
# apply multiplication
for col in cols_to_multiply:
    Bonfcorrect_Ttest_1a_comb[col] = Bonfcorrect_Ttest_1a_comb[col] * 3

Bonfcorrect_Ttest_1a_comb.to_csv('paired_sample_Ttest_Bonfcorrection_COMBINED_1A.csv')

