import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import shapiro
from study2_5Closest import X, y
from T_psychometric import metrics_psy

X, y = X.copy(), y.copy()

X1, X_temp, y1, y_temp = train_test_split(X, y, test_size=0.66, random_state=42)
X2, X3, y2, y3 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

parameter_values = {'SVR-RBF': [100],
                    'Ridge Regression': [10]}

models = {'SVR-RBF': SVR(kernel='rbf'),
          'Ridge Regression': Ridge()}


metrics_list = []
prediction_list = []
datasets_X = [X1, X2, X3]
datasets_y = [y1, y2, y3]
subset_names = ['Subset1', 'Subset2', 'Subset3']

for subset_name, X_subset, y_subset in zip(subset_names, datasets_X, datasets_y):
    subset_results = {}
    subset_predictions = {}
    for model_name, model in tqdm(models.items()):
        if model_name == 'SVR-RBF':
            model.set_params(C=100)
        elif model_name == 'Ridge Regression':
            model.set_params(alpha=10)

        model.fit(X_subset, y_subset)
        y_pred = model.predict(X_subset)
        residuals = y_subset - y_pred
        subset_predictions[f'{model_name} Predictions'] = y_pred

        # Store metrics
        mse = mean_squared_error(y_subset, y_pred)
        r2 = model.score(X_subset, y_subset)
        metrics_list.append({'Subset': subset_name, 'Model': model_name, 'RMSE_5close': np.mean(np.sqrt(mse)), 'R2_5close': np.mean(r2)})
        prediction_list.append({'Subset': subset_name, 'Model': model_name, 'y_pred_5close': y_pred})


metrics_5close = pd.DataFrame(metrics_list)
prediction_5close = pd.DataFrame(prediction_list)

# ------------------------------------------------- difference 1close & psy --------------------------------------------

diffR2_RBF_psy = metrics_5close[metrics_5close['Model'] == 'SVR-RBF']['R2_5close'] - metrics_psy[metrics_psy['Model'] == 'SVR-RBF']['R2_psy']
diffRMSE_RBF_psy = metrics_5close[metrics_5close['Model'] == 'SVR-RBF']['RMSE_5close'] - metrics_psy[metrics_psy['Model'] == 'SVR-RBF']['RMSE_psy']

diffR2_Ridge_psy = metrics_5close[metrics_5close['Model'] == 'Ridge Regression']['R2_5close'] - metrics_psy[metrics_psy['Model'] == 'Ridge Regression']['R2_psy']
diffRMSE_Ridge_psy = metrics_5close[metrics_5close['Model'] == 'Ridge Regression']['RMSE_5close'] - metrics_psy[metrics_psy['Model'] == 'Ridge Regression']['RMSE_psy']

# ---------------------------------------- Performing Shapiro-Wilk test for normality ---------------------------------

diffR2_RBF_psy_stat, diffR2_RBF_psy_p = shapiro(diffR2_RBF_psy)
diffRMSE_RBF_psy_stat, diffRMSE_RBF_psy_p = shapiro(diffRMSE_RBF_psy)

diffR2_Ridge_psy_stat, diffR2_Ridge_psy_p = shapiro(diffR2_Ridge_psy)
diffRMSE_Ridge_psy_stat, diffRMSE_Ridge_psy_p = shapiro(diffRMSE_Ridge_psy)

# save the results of the Shapiro-Wilk test in a dataframe
shapiro_test5close = pd.DataFrame({
    'model': ['SVR-RBF', 'Ridge Regression'],
    'statistic_R2': [diffR2_RBF_psy_stat, diffR2_Ridge_psy_stat],
    'p_value_R2': [diffR2_RBF_psy_p, diffR2_Ridge_psy_p],
    'statistic_RMSE': [diffRMSE_RBF_psy_stat, diffRMSE_Ridge_psy_stat],
    'p_value_RMSE': [diffRMSE_RBF_psy_p, diffRMSE_Ridge_psy_p]})


# ---------------------------------------- Performing paired sample t-tests ----------------------------------------

RBF_psy_t2_statistic_r2, RBF_psy_p2_value_r2 = stats.ttest_rel(metrics_5close[metrics_5close['Model'] == 'SVR-RBF']['R2_5close'], metrics_psy[metrics_psy['Model'] == 'SVR-RBF']['R2_psy'])
RBF_psy_t2_statistic_rmse, RBF_psy_p2_value_rmse = stats.ttest_rel(metrics_5close[metrics_5close['Model'] == 'SVR-RBF']['RMSE_5close'], metrics_psy[metrics_psy['Model'] == 'SVR-RBF']['RMSE_psy'])
Ridge_psy_t2_statistic_r2, Ridge_psy_p2_value_r2 = stats.ttest_rel(metrics_5close[metrics_5close['Model'] == 'Ridge Regression']['R2_5close'], metrics_psy[metrics_psy['Model'] == 'Ridge Regression']['R2_psy'])
Ridge_psy_t2_statistic_rmse, Ridge_psy_p2_value_rmse = stats.ttest_rel(metrics_5close[metrics_5close['Model'] == 'Ridge Regression']['RMSE_5close'], metrics_psy[metrics_psy['Model'] == 'Ridge Regression']['RMSE_psy'])

# save the results of the paired sample t-tests in a dataframe
t_test_5close = pd.DataFrame({
    'model': ['SVR-RBF', 'Ridge Regression'],
    'statistic_R2': [RBF_psy_t2_statistic_r2, Ridge_psy_t2_statistic_r2],
    'p_value_R2': [RBF_psy_p2_value_r2, Ridge_psy_p2_value_r2],
    'statistic_RMSE': [RBF_psy_t2_statistic_rmse, Ridge_psy_t2_statistic_rmse],
    'p_value_RMSE': [RBF_psy_p2_value_rmse, Ridge_psy_p2_value_rmse]})

# Bonferroni correction

Bonfcorrect_Ttest_5close = t_test_5close.copy()

# columns to be multiplied
cols_to_multiply = ['p_value_R2', 'p_value_RMSE']
# apply multiplication
for col in cols_to_multiply:
    Bonfcorrect_Ttest_5close[col] = Bonfcorrect_Ttest_5close[col] * 3
