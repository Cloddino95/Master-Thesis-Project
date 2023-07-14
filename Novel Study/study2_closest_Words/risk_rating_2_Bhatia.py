from gensim.models import KeyedVectors
from Dataset import risk_ratings_2
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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import learning_curve


model = gensim.models.KeyedVectors.load(
    '/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

risk_source_name_2 = risk_ratings_2.iloc[0].values.tolist()
risk_source_name_2 = [word.replace(' ', '_') for word in risk_source_name_2]

# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_2 = [word for word in risk_source_name_2 if word in vocab]

omitted_word_2 = [word for word in risk_source_name_2 if word not in valid_risk_source_name_2]

# noinspection PyUnresolvedReferences
risk_source_vector_2 = [model[word] for word in valid_risk_source_name_2]

dict_risk_source_vectors_2 = dict(zip(valid_risk_source_name_2, risk_source_vector_2))


def get_vector_for_source_i(a):
    source_name = valid_risk_source_name_2[a]
    return dict_risk_source_vectors_2[source_name]


xi_vectors_2 = []
for i in range(len(valid_risk_source_name_2)):
    xi = get_vector_for_source_i(i)
    xi_vectors_2.append(xi)

mat_xi_300dim_2 = np.array(xi_vectors_2)

data_300dim_2 = mat_xi_300dim_2.copy()
data_300dim_2 = pd.DataFrame(data_300dim_2, index=valid_risk_source_name_2)

risk_ratings_2_small = risk_ratings_2.copy()
risk_ratings_2_small.columns = risk_source_name_2
risk_ratings_2_small = risk_ratings_2_small.drop(risk_ratings_2_small.index[0])
risk_ratings_2_small = risk_ratings_2_small.drop(omitted_word_2, axis=1)
risk_ratings_2_small = risk_ratings_2_small.T
risk_ratings_2_small = risk_ratings_2_small.dropna(axis=1, how='all')
risk_ratings_2_small = risk_ratings_2_small.astype(float)

mean_rows = risk_ratings_2_small.mean(axis=1)
data_300dim_2.insert(0, "mean_ratings", mean_rows)

X = data_300dim_2.drop('mean_ratings', axis=1)
y = data_300dim_2['mean_ratings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Ridge_ = Ridge().set_params(alpha=4)
Ridge_.fit(X_train, y_train)
Ridge_pred_test = Ridge_.predict(X_test)
Ridge_pred_train = Ridge_.predict(X_train)

SVR_RBF = SVR(kernel='rbf').set_params(C=100)
SVR_RBF.fit(X_train, y_train)
SVR_RBF_pred_test = SVR_RBF.predict(X_test)
SVR_RBF_pred_train = SVR_RBF.predict(X_train)


data_SVR = {"names": y_test.index.tolist(), "(SVR-RBF) predicted values": pd.Series(SVR_RBF_pred_test.flatten(), index=y_test.index)}
SVR_RBF_pred_test2_df = pd.DataFrame(data_SVR)
data_Ridge = {"(Ridge) predicted values": pd.Series(Ridge_pred_test.flatten(), index=y_test.index), "names": y_test.index.tolist()}
Ridge_pred_test2_df = pd.DataFrame(data_Ridge)
data_y = {"Actual values": y_test, "(Y - SVR-RBF) Residuals": y_test - pd.Series(SVR_RBF_pred_test.flatten(), index=y_test.index), "(Y - Ridge) Residuals": y_test - pd.Series(Ridge_pred_test.flatten(), index=y_test.index)}
data_y_df = pd.DataFrame(data_y)
prediction_Bha = pd.concat([SVR_RBF_pred_test2_df, Ridge_pred_test2_df["(Ridge) predicted values"], data_y_df], axis=1)


# Metrics for SVR-RBF
SVR_RBF_mse_test2 = mean_squared_error(y_test, SVR_RBF_pred_test)
SVR_RBF_mse_train2 = mean_squared_error(y_train, SVR_RBF_pred_train)
SVR_RBF_r2_test2 = SVR_RBF.score(X_test, y_test)
SVR_RBF_r2_train2 = SVR_RBF.score(X_train, y_train)

# Metrics for Ridge Regression
Ridge_mse_test2 = mean_squared_error(y_test, Ridge_pred_test)
Ridge_mse_train2 = mean_squared_error(y_train, Ridge_pred_train)
Ridge_r2_test2 = Ridge_.score(X_test, y_test)
Ridge_r2_train2 = Ridge_.score(X_train, y_train)

results_list2_ridge = {'R2 test': Ridge_r2_test2, 'RMSE test': np.sqrt(Ridge_mse_test2), 'R2 training': Ridge_r2_train2, 'RMSE training': np.sqrt(Ridge_mse_train2)}
results_list2_SVR = {'R2 test': SVR_RBF_r2_test2, 'RMSE test': np.sqrt(SVR_RBF_mse_test2), 'R2 training': SVR_RBF_r2_train2, 'RMSE training': np.sqrt(SVR_RBF_mse_train2)}
metrics_Bha = pd.DataFrame([results_list2_ridge, results_list2_SVR])
metrics_Bha.index = ['Ridge', 'SVR-RBF']


# ----------------------------------------------------------------------------------------------------------------------
# CROSS VALIDATION NEEDED TO ENSURE GENERALISATION AND TO PERFROM THE BELOW HYPERPARAMETER TUNING
parameter_values2 = {'SVR-RBF': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                     'Ridge Regression': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

models2 = {'SVR-RBF': SVR(kernel='rbf'), 'Ridge Regression': Ridge()}

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

results_list_2 = []

for model_name, model in tqdm(models2.items()):
    if model_name == 'Ridge Regression':
        for k in parameter_values2[model_name]:
            model.set_params(alpha=k)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = - cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            results_list_2.append({'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(r2_scores),
                                   'std_score_R2': np.std(r2_scores),
                                   'mean_score_RMSE': np.mean(np.sqrt(mse_scores)), 'std_score_RMSE': np.std(np.sqrt(mse_scores))})
    else:
        for C in tqdm(parameter_values2[model_name]):
            model.set_params(C=C)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = - cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            results_list_2.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores),
                 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)), 'std_score_RMSE': np.std(np.sqrt(mse_scores))})

results_df_2_Bhatia = pd.DataFrame(results_list_2, index=range(len(results_list_2)))

tot_results_df_2_Bhatia = results_df_2_Bhatia.drop('parameter', axis=1).groupby('model').mean()
# Save the results dataframe to a csv file
# results_df_2_Bhatia.to_csv('results_df_2_Bhatia.csv', index=False)

# --------------------------------------------------Ridge---------------------------------------------------------------

"""The ***fill_between function*** is used to indicate the standard deviation of the RMSE score at each meta-parameter value, 
giving you an idea of the variability of the RMSE at each point."""

# ----------------------------------------ALPHA VALUES---------------------------------------------------------------
"""The penalization coefficient (alpha) in Ridge Regression serves to control the complexity of the model.
In practice, alpha is often chosen through cross-validation. This involves running Ridge Regression with different alpha values 
and seeing which one results in the lowest cross-validated error (or in your case, RMSE).
This is exactly what you are doing in your code below. Based on the plot generated from the code I provided, you would choose 
the alpha that gives the lowest mean RMSE.

!!!!In some cases, it might be better to choose a slightly higher alpha even if it doesn't give the absolute lowest RMSE if the 
standard deviation of the RMSE is significantly lower, as this would suggest that the model performs more consistently across 
different splits of the data.!!!!"""

# Select the Ridge Regression results
ridge_results = results_df_2_Bhatia[results_df_2_Bhatia['model'] == 'Ridge Regression']

# Set up the figure
plt.figure(figsize=(12, 8))
# Plot the mean RMSE score and fill the surrounding area with one standard deviation range
plt.plot(ridge_results['parameter'], ridge_results['mean_score_RMSE'], label='Mean RMSE', color='navy')
plt.fill_between(ridge_results['parameter'],
                 ridge_results['mean_score_RMSE'] - ridge_results['std_score_RMSE'],
                 ridge_results['mean_score_RMSE'] + ridge_results['std_score_RMSE'],
                 color='skyblue', alpha=0.5,
                 label='Mean RMSE ± 1 std. dev.')
# Define the x-axis ticks
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))  # Set major ticks at increments of 0.2
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # Set minor ticks at increments of 0.1
# Styling and labelling
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Light grid lines for better reading
plt.title('RMSE as a function of Ridge Regression alpha parameter', fontsize=15)
plt.xlabel('Ridge Alpha Parameter', fontsize=12)
plt.ylabel('Mean RMSE', fontsize=12)
plt.legend(fontsize=10)
sns.despine()  # Remove the top and right spines for a cleaner look
# Show the plot
plt.savefig('Ridge_alpha_Bha.png')
plt.show()


# --------------------------------------------PATH DIAGRAM-------------------------------------------------------------
# It shows how the coefficients of the Ridge regression model change as the penalty parameter alpha increases.

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Initialize an array to hold the coefficients
coefs = []

# For each alpha value
for a in alphas:
    # Create and fit the Ridge model
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    # Store the coefficients
    coefs.append(ridge.coef_)

# Create the plot
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(alphas, coefs)
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
xticks = np.arange(min(alphas), max(alphas)+0.1, 0.1)
plt.xticks(xticks)
plt.axis('tight')
plt.savefig('Ridge_Path_Diagram_Bha.png')
plt.show()


# --------------------------------------------LEARNING CURVES-----------------------------------------------------------
"""This script below will first define the plot_learning_curve function. Then it will loop over the alpha values for the Ridge
 regression model specified in parameter_values2['Ridge Regression']. For each alpha value, it will create a Ridge regression 
 estimator with that alpha value, generate a learning curve, and display it. At the end, it will show all the learning curve plots."""

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, scoring=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc="best")
    return plt


# Iterate over the Ridge alpha values
ridge_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for alpha_value in ridge_alphas:
    # Title for the plot
    title = f"Learning Curves (Ridge Regression, alpha={alpha_value})"
    # Ridge estimator for the current alpha value
    estimator = Ridge(alpha=alpha_value)
    # Plot the learning curve
    plot_learning_curve(estimator, title, X, y, cv=kf)
    plt.savefig(f"LearnCurve_Ridge_alpha{alpha_value}.png")
    plt.show()

# ------------------------------------- same analysis but for SVR-RBF --------------------------------------------------


# -------------------------------------------- C value -----------------------------------------------------------
"""With this plot, you can observe how the RMSE changes with different C values (the penalty parameter of the error term) for SVR 
with the RBF kernel. A smaller RMSE indicates a better fit of the model to the data."""

# Get SVR-RBF results
svr_results = results_df_2_Bhatia[results_df_2_Bhatia['model'] == 'SVR-RBF']

# Set up the figure
plt.figure(figsize=(12, 8))

# Plot the mean RMSE score and fill the surrounding area with one standard deviation range
plt.plot(svr_results['parameter'], svr_results['mean_score_RMSE'], label='Mean RMSE', color='navy')
plt.fill_between(svr_results['parameter'],
                 svr_results['mean_score_RMSE'] - svr_results['std_score_RMSE'],
                 svr_results['mean_score_RMSE'] + svr_results['std_score_RMSE'],
                 color='skyblue', alpha=0.5,
                 label='Mean RMSE ± 1 std. dev.')
# Define the x-axis ticks
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  # Set major ticks at increments of 100
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))  # Set minor ticks at increments of 50
# Styling and labelling
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Light grid lines for better reading
plt.title('RMSE as a function of SVR-RBF C parameter', fontsize=15)
plt.xlabel('SVR-RBF C Parameter', fontsize=12)
plt.ylabel('Mean RMSE', fontsize=12)
plt.legend(fontsize=10)
sns.despine()  # Remove the top and right spines for a cleaner look
plt.savefig('SVR_C.png')
plt.show()

# --------------------------------------------LEARNING CURVES-----------------------------------------------------------

# C values you're interested in
svr_C_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# For each C value
for C_value in svr_C_values:
    title = f"Learning Curves (SVR-RBF, C={C_value})"
    estimator = SVR(kernel='rbf', C=C_value)

    # Plot the learning curve using RMSE
    plot_learning_curve(estimator, title, X, y, cv=kf, n_jobs=None)
    plt.savefig(f"LearnCurve_SVR_C{C_value}.png")
    plt.show()
