from psychometric_2 import X, y, risk_source_name_2
# from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
# import gensim
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error  #, make_scorer
# from tqdm import tqdm
from sklearn.model_selection import train_test_split

X = X.copy()
y = y.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train_size = int(0.8 * len(X))
# X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

parameter_values = {'SVR-RBF': [100],
                    'Ridge Regression': [4]}

models = {'SVR-RBF': SVR(kernel='rbf'),
          'Ridge Regression': Ridge()}


Ridge_psy2 = Ridge().set_params(alpha=4)
Ridge_psy2.fit(X_train, y_train)
Ridge_pred_test2 = Ridge_psy2.predict(X_test)
Ridge_pred_train2 = Ridge_psy2.predict(X_train)

SVR_RBF_psy2 = SVR(kernel='rbf').set_params(C=100)
SVR_RBF_psy2.fit(X_train, y_train)
SVR_RBF_pred_test2 = SVR_RBF_psy2.predict(X_test)
SVR_RBF_pred_train2 = SVR_RBF_psy2.predict(X_train)


data_SVR = {"names": y_test.index.tolist(), "(SVR-RBF) predicted values": pd.Series(SVR_RBF_pred_test2.flatten(), index=y_test.index)}
SVR_RBF_pred_test2_df = pd.DataFrame(data_SVR)
data_Ridge = {"(Ridge) predicted values": pd.Series(Ridge_pred_test2.flatten(), index=y_test.index), "names": y_test.index.tolist()}
Ridge_pred_test2_df = pd.DataFrame(data_Ridge)
data_y = {"Actual values": y_test, "(Y - SVR-RBF) Residuals": y_test - pd.Series(SVR_RBF_pred_test2.flatten(), index=y_test.index), "(Y - Ridge) Residuals": y_test - pd.Series(Ridge_pred_test2.flatten(), index=y_test.index)}
data_y_df = pd.DataFrame(data_y)
prediction_psy2 = pd.concat([SVR_RBF_pred_test2_df, Ridge_pred_test2_df["(Ridge) predicted values"], data_y_df], axis=1)


# Metrics for SVR-RBF
SVR_RBF_mse_test2 = mean_squared_error(y_test, SVR_RBF_pred_test2)
SVR_RBF_mse_train2 = mean_squared_error(y_train, SVR_RBF_pred_train2)
SVR_RBF_r2_test2 = SVR_RBF_psy2.score(X_test, y_test)
SVR_RBF_r2_train2 = SVR_RBF_psy2.score(X_train, y_train)

# Metrics for Ridge Regression
Ridge_mse_test2 = mean_squared_error(y_test, Ridge_pred_test2)
Ridge_mse_train2 = mean_squared_error(y_train, Ridge_pred_train2)
Ridge_r2_test2 = Ridge_psy2.score(X_test, y_test)
Ridge_r2_train2 = Ridge_psy2.score(X_train, y_train)

results_list2_ridge = {'R2 test': Ridge_r2_test2, 'RMSE test': np.sqrt(Ridge_mse_test2), 'R2 training': Ridge_r2_train2, 'RMSE training': np.sqrt(Ridge_mse_train2)}
results_list2_SVR = {'R2 test': SVR_RBF_r2_test2, 'RMSE test': np.sqrt(SVR_RBF_mse_test2), 'R2 training': SVR_RBF_r2_train2, 'RMSE training': np.sqrt(SVR_RBF_mse_train2)}
metrics_psy2 = pd.DataFrame([results_list2_ridge, results_list2_SVR])
metrics_psy2.index = ['Ridge', 'SVR-RBF']
