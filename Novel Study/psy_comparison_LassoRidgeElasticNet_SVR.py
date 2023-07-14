"""
let's first assess which parameter are the best for elastic net regression
then let's calculate the RMSE and R2 for the psy approach using ridge lasso elastic net and SVR to then choose the best model (one of the three plus SVR)
"""
from psychometric_2 import X, y
# from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import gensim
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from tqdm import tqdm

X = X.copy()
y = y.copy()

parameter_values = {
    'SVR-RBF': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'Lasso Regression': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'Ridge Regression': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'ElasticNet Regression alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'ElasticNet Regression l1_ratio': [0.5]
}

models = {'SVR-RBF': SVR(kernel='rbf'),
          'Lasso Regression': Lasso(),
          'Ridge Regression': Ridge(),
          'ElasticNet Regression': ElasticNet(alpha=1.0, l1_ratio=0.5)}

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

results_list_2 = []

for model_name, model in tqdm(models.items()):
    if model_name == 'ElasticNet Regression':
        for k in parameter_values['ElasticNet Regression alpha']:
            model.set_params(alpha=k, l1_ratio=0.5)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = - cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            results_list_2.append({'model': model_name, 'parameter': f'alpha={k}, l1_ratio=0.5', 'mean_score_R2': np.mean(r2_scores),
                                   'std_score_R2': np.std(r2_scores),
                                   'mean_score_RMSE': np.mean(np.sqrt(mse_scores)), 'std_score_RMSE': np.std(np.sqrt(mse_scores))})
    else:
        for C in tqdm(parameter_values[model_name]):
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = - cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            results_list_2.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores),
                 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)), 'std_score_RMSE': np.std(np.sqrt(mse_scores))})

results_df_2_Bhatia = pd.DataFrame(results_list_2, index=range(len(results_list_2)))

"""
Based on the above analysis we can conclude that Ridge performs better than Lasso and ElasticNet. This support my thoery
that applying variable selection in the context of word embedding can be detrimental since each dimension carry a distance (so an information)
for its semantic meaning. so eliminating a dimension even if reduce complexity, it also reduce the semantic information. 
This is not the case since this is a normal linear regression but since i have to choose a model to compare the two approaches
i will then choose the Ridge Regression and SVR so that i have both a linear and non-linear model to compare with the semantic approach.
"""